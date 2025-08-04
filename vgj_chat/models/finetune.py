"""LoRA fine-tuning helper."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# dataset built by ``vgj_chat.data.dataset.build_auto_dataset``
# automatically generated Q&A pairs
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")
CHECKPOINT_DIR = Path("data/lora-vgj-checkpoint")
MODEL_CACHE = Path("data/model_cache")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_PER_GPU = 4
GRAD_ACC_STEPS = 4
LOG_STEPS = 1
EVAL_STEPS = 1
PATIENCE = 3
EPOCHS = 10
LR = 2e-4


def run_finetune() -> None:
    if CHECKPOINT_DIR.exists():
        print(f"{CHECKPOINT_DIR} exists; skipping fine-tune")
        return
    hf_token = os.getenv("VGJ_HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, use_fast=True, token=hf_token, cache_dir=MODEL_CACHE
    )
    tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map={"": 0},
            torch_dtype=torch.float16,
            token=hf_token,
            cache_dir=MODEL_CACHE,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            token=hf_token,
            cache_dir=MODEL_CACHE,
        )
    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    def to_chat(ex):
        return {
            "text": f"<s>[INST] {ex['input'].strip()} [/INST] {ex['output'].strip()} </s>"
        }

    dataset = load_dataset("json", data_files=str(AUTO_QA_JL), split="train").map(
        to_chat, remove_columns=["input", "output"]
    )
    train_idx, eval_idx = train_test_split(
        list(range(len(dataset))), test_size=0.1, random_state=42
    )
    train_set = dataset.select(train_idx)
    eval_set = dataset.select(eval_idx)
    train_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=BATCH_PER_GPU,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=PATIENCE, early_stopping_threshold=0.0
            )
        ],
    )
    trainer.train()
    model.save_pretrained(CHECKPOINT_DIR)
    tok.save_pretrained(CHECKPOINT_DIR)
    print(f"LoRA adapter + tokenizer saved to → {CHECKPOINT_DIR}")
