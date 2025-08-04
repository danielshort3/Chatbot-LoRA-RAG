"""LoRA fine-tuning helper."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
try:  # transformers is stubbed in tests
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - fallback for minimal stubs
    class TrainerCallback:  # type: ignore
        pass

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
    # disable cache to avoid incompatibilities with gradient checkpointing
    model.config.use_cache = False
    # show which parameters will receive updates
    model.print_trainable_parameters()

    class GradDebugCallback(TrainerCallback):
        """Callback to print gradient norms and track parameter updates."""

        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model
            # snapshot of initial parameters to verify updates at the end
            self._initial = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }

        def on_backward_end(self, args, state, control, **kwargs):
            norms: dict[str, float] = {}
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    norms[name] = float(p.grad.detach().norm())
            total = sum(g ** 2 for g in norms.values()) ** 0.5
            print(f"[GradDebug] step {state.global_step} grad_norm={total:.6f}")
            # print a few individual layer norms to confirm non-zero grads
            for n, g in list(norms.items())[:3]:
                print(f"[GradDebug]   {n} grad_norm={g:.6f}")
            return control

        def on_train_end(self, args, state, control, **kwargs):
            updated = False
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if not torch.equal(p.detach(), self._initial[name]):
                    updated = True
                    delta = (p.detach() - self._initial[name]).abs().max().item()
                    print(
                        f"[GradDebug] param {name} changed; max_abs_diff={delta:.6e}"
                    )
                    break
            if not updated:
                print("[GradDebug] trainable parameters did not change!")
            return control

    dataset = load_dataset("json", data_files=str(AUTO_QA_JL))["train"].rename_columns(
        {"input": "prompt", "output": "response"}
    )
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_set, eval_set = split["train"], split["test"]
    print(
        f"[Dataset] train_examples={len(train_set)} eval_examples={len(eval_set)}"
    )
    first = train_set[0]
    print(
        f"[Dataset] sample prompt={first['prompt']!r} response={first['response']!r}"
    )

    def format_example(ex: dict) -> str:
        return tok.apply_chat_template(
            [
                {"role": "user", "content": ex["prompt"].strip()},
                {"role": "assistant", "content": ex["response"].strip()},
            ],
            tokenize=False,
        )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
        fp16=False,
        bf16=use_bf16,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        formatting_func=format_example,
        train_on_inputs=False,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=PATIENCE, early_stopping_threshold=0.0
            ),
            GradDebugCallback(model),
        ],
    )
    # sanity check that labels contain non-masked tokens
    batch = next(iter(trainer.get_train_dataloader()))
    non_masked = int((batch["labels"] != -100).sum())
    print(
        f"[SanityCheck] first batch total_tokens={batch['labels'].numel()} non_masked={non_masked}"
    )
    trainer.train()
    model.save_pretrained(CHECKPOINT_DIR)
    tok.save_pretrained(CHECKPOINT_DIR)
    print(f"LoRA adapter + tokenizer saved to → {CHECKPOINT_DIR}")
