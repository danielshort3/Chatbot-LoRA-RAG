"""Standalone LoRA fine-tuning with extensive diagnostics.

Example:
    python scripts/finetune.py --data data/qa.jsonl --output-dir data/lora-vgj-checkpoint
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable

import random
import numpy as np
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

MODEL_CACHE = Path("data/model_cache")


# ---------------------------------------------------------------------------
# Argument parsing and configuration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning script")
    parser.add_argument("--data", type=str, required=True, help="JSONL file with Q&A pairs")
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output-dir", type=str, default="data/lora-vgj-checkpoint")
    parser.add_argument("--prompt-field", type=str, default="input")
    parser.add_argument("--response-field", type=str, default="output")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, help="Optional YAML config file")
    args = parser.parse_args()

    # load YAML if provided – CLI values take precedence
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if not hasattr(args, key.replace('-', '_')):
                raise ValueError(f"Unknown config key: {key}")
            dest = key.replace('-', '_')
            if getattr(args, dest) == parser.get_default(dest):
                setattr(args, dest, value)
    return args


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

def load_and_tokenize(
    data_path: str,
    tokenizer: AutoTokenizer,
    prompt_field: str,
    response_field: str,
    seed: int,
) -> Dict[str, 'datasets.Dataset']:
    if not Path(data_path).exists():
        raise FileNotFoundError(data_path)
    ds = load_dataset("json", data_files=data_path)["train"]
    missing = [f for f in (prompt_field, response_field) if f not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset missing fields: {missing}")

    def _format(example: Dict[str, str]) -> str:
        """Apply chat template for a single example."""
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example[prompt_field].strip()},
                {"role": "assistant", "content": example[response_field].strip()},
            ],
            tokenize=False,
        )

    def _tokenize(example: Dict[str, str]) -> Dict[str, Iterable[int]]:
        text = _format(example)
        toks = tokenizer(text, truncation=False)
        if len(toks["input_ids"]) > tokenizer.model_max_length:
            raise ValueError(
                f"Sequence length {len(toks['input_ids'])} exceeds model limit {tokenizer.model_max_length}"
            )
        toks = tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=0.1, seed=seed)
    tokenized = {k: v.map(_tokenize, remove_columns=v.column_names) for k, v in split.items()}
    return tokenized


# ---------------------------------------------------------------------------
# Diagnostics callback
# ---------------------------------------------------------------------------

class DiagnosticsCallback(TrainerCallback):
    """Report losses, gradient norms, GPU memory and LR schedule."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            ppl = math.exp(logs["loss"])
            print(f"[step {step}] train_loss={logs['loss']:.4f} ppl={ppl:.2f}")
        if "eval_loss" in logs:
            eval_ppl = math.exp(logs["eval_loss"])
            print(f"[step {step}] val_loss={logs['eval_loss']:.4f} val_ppl={eval_ppl:.2f}")
        if "learning_rate" in logs:
            print(f"[step {step}] lr={logs['learning_rate']:.6e}")
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2
            print(f"[step {step}] cuda_mem={mem:.0f}MB")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0 or state.global_step == 0:
            return
        norms = []
        zero_grad = []
        for name, p in self.model.named_parameters():
            if "lora" in name.lower() and p.requires_grad:
                if p.grad is None or p.grad.abs().sum() == 0:
                    zero_grad.append(name)
                    continue
                norms.append(p.grad.detach().norm().item())
        if norms:
            total = math.sqrt(sum(n ** 2 for n in norms))
            print(f"[step {state.global_step}] lora_grad_norm={total:.6f}")
        if zero_grad:
            print(
                f"[step {state.global_step}] zero-grad params: {', '.join(zero_grad[:3])}"
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    # Deterministic behaviour
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    token = os.getenv("VGJ_HF_TOKEN")
    if token:
        login(token=token)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, token=token, cache_dir=MODEL_CACHE
    )
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
            token=token,
            cache_dir=MODEL_CACHE,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, token=token, cache_dir=MODEL_CACHE
        )
    # Some tokenizers report an extremely large model_max_length (e.g. 1e30) to
    # signify "no limit", which overflows the Rust implementation used by
    # `tokenizers` when passed as `max_length`.  Clamp the tokenizer length to
    # the actual context window defined in the model config to avoid
    # `OverflowError: int too big to convert` during tokenization.
    max_pos = getattr(base.config, "max_position_embeddings", None)
    if max_pos and max_pos > 0:
        tokenizer.model_max_length = min(tokenizer.model_max_length, max_pos)

    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.config.use_cache = False
    model.print_trainable_parameters()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters detected")
    if not any("lora" in n.lower() for n in trainable):
        raise ValueError("LoRA parameters are frozen")

    datasets = load_and_tokenize(
        args.data, tokenizer, args.prompt_field, args.response_field, args.seed
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
        callbacks=[DiagnosticsCallback(model)],
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅  LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
