"""LoRA/QLoRA fine‑tuning helper.

This script follows the recommended recipe for adapting
``openai/gpt-oss-20b`` into the Visit Grand Junction travel‑agent model. It
mirrors the steps described in the rough guide:

* load the gpt‑oss tokenizer and register optional special tokens
  (``<CONTEXT>``, ``</CONTEXT>``) that can be used when injecting retrieved
  chunks during RAG inference;
* quantise the base model to 4‑bit weights and apply a LoRA adapter to
  the major linear layers (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``,
  ``gate_proj``, ``up_proj`` and ``down_proj``) with rank 16 and zero
  dropout as suggested by Unsloth;
* train with a small per‑device batch size and gradient accumulation to
  reach an effective batch size of ~16 and enable gradient
  checkpointing to reduce memory usage.
"""

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
except Exception:  # pragma: no cover – fallback for minimal stubs

    class TrainerCallback:  # type: ignore
        pass


from trl import SFTConfig, SFTTrainer  # NEW import

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_MODEL = "openai/gpt-oss-20b"
# dataset built by ``vgj_chat.data.dataset.build_auto_dataset`` – Q&A pairs
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")
CHECKPOINT_DIR = Path("data/lora-vgj-checkpoint")
MODEL_CACHE = Path("data/model_cache")

# LoRA/QLoRA hyper‑parameters
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# training hyper‑parameters
BATCH_PER_GPU = 2
GRAD_ACC_STEPS = 8
LOG_STEPS = 10
EVAL_STEPS = 50
PATIENCE = 3
EPOCHS = 2
LR = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# optional special tokens for RAG context delimitation
SPECIAL_TOKENS = {"additional_special_tokens": ["<CONTEXT>", "</CONTEXT>"]}


# --------------------------------------------------------------------------- #
# Fine‑tuning entry point
# --------------------------------------------------------------------------- #
def run_finetune() -> None:
    if CHECKPOINT_DIR.exists():
        print(f"{CHECKPOINT_DIR} exists; skipping fine‑tune")
        return

    hf_token = os.getenv("VGJ_HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------- tokenizer ------------------------------------ #
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, use_fast=False, token=hf_token, cache_dir=MODEL_CACHE
    )
    tok.add_special_tokens(SPECIAL_TOKENS)
    tok.pad_token = tok.eos_token

    # ----------------------- base model ----------------------------------- #
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

    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base)

    # ----------------------- LoRA adapter --------------------------------- #
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base, lora_cfg)
    model.config.use_cache = False  # avoid grad‑checkpoint/cache clash
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    # ----------------------- grad‑debug callback -------------------------- #
    class GradDebugCallback(TrainerCallback):
        """Print gradient norms each step and confirm params updated."""

        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model
            self._initial = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }

        def on_backward_end(self, args, state, control, **kwargs):
            norms: dict[str, float] = {
                n: float(p.grad.detach().norm())
                for n, p in self.model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            total = sum(g**2 for g in norms.values()) ** 0.5
            print(f"[GradDebug] step {state.global_step} grad_norm={total:.6f}")
            for n, g in list(norms.items())[:3]:
                print(f"[GradDebug]   {n} grad_norm={g:.6f}")
            return control

        def on_train_end(self, args, state, control, **kwargs):
            for name, p in self.model.named_parameters():
                if p.requires_grad and not torch.equal(p.detach(), self._initial[name]):
                    delta = (p.detach() - self._initial[name]).abs().max().item()
                    print(f"[GradDebug] param {name} changed; max_abs_diff={delta:.6e}")
                    break
            else:
                print("[GradDebug] trainable parameters did not change!")
            return control

    # ----------------------- dataset -------------------------------------- #
    dataset = load_dataset("json", data_files=str(AUTO_QA_JL))["train"]
    if {"input", "output"}.issubset(set(dataset.column_names)):
        dataset = dataset.rename_columns({"input": "prompt", "output": "response"})

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_set, eval_set = split["train"], split["test"]

    print(f"[Dataset] train_examples={len(train_set)}  eval_examples={len(eval_set)}")
    first = train_set[0]
    if "messages" in first:
        print(f"[Dataset] sample messages={first['messages']!r}")
    else:
        print(
            f"[Dataset] sample prompt={first['prompt']!r}  response={first['response']!r}"
        )

    def format_example(ex: dict) -> str:
        if "messages" in ex:
            return tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok.apply_chat_template(
            [
                {"role": "user", "content": ex["prompt"].strip()},
                {"role": "assistant", "content": ex["response"].strip()},
            ],
            tokenize=False,
        )

    # ----------------------- training arguments --------------------------- #
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=BATCH_PER_GPU,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
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

    # ----------------------- SFT config (key fix) ------------------------- #
    sft_cfg = SFTConfig(
        completion_only_loss=False,  # disable so formatter is allowed
        dataset_text_field=None,  # let trainer use formatter output
    )

    # ----------------------- trainer -------------------------------------- #
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        formatting_func=format_example,
        config=sft_cfg,  # NEW
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=PATIENCE, early_stopping_threshold=0.0
            ),
            GradDebugCallback(model),
        ],
    )

    # sanity check – non‑masked label tokens
    batch = next(iter(trainer.get_train_dataloader()))
    non_masked = int((batch["labels"] != -100).sum())
    print(
        f"[SanityCheck] first batch total_tokens={batch['labels'].numel()} "
        f"non_masked={non_masked}"
    )

    # ----------------------- train & save --------------------------------- #
    trainer.train()
    model.save_pretrained(CHECKPOINT_DIR)
    tok.save_pretrained(CHECKPOINT_DIR)
    print(f"✅  LoRA adapter + tokenizer saved to → {CHECKPOINT_DIR}")
