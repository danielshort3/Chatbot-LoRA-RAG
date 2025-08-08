import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE = "openai/gpt-oss-20b"
DEFAULT_LORA = "data/lora-vgj-checkpoint"
DEFAULT_OUT = "data/gpt-oss-20b-merged-4bit"
MODEL_CACHE = Path("data/model_cache")

# must match special tokens used during fine-tuning
SPECIAL_TOKENS = {"additional_special_tokens": ["<CONTEXT>", "</CONTEXT>"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--lora-dir", default=DEFAULT_LORA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    token = os.getenv("VGJ_HF_TOKEN")
    if token:
        login(token=token)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"{out_dir} exists; skipping merge")
        return

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, token=token, cache_dir=MODEL_CACHE
    )
    tok.add_special_tokens(SPECIAL_TOKENS)
    tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=qconfig,
        token=token,
        cache_dir=MODEL_CACHE,
    )
    base.resize_token_embeddings(len(tok))

    lora = PeftModel.from_pretrained(base, args.lora_dir)
    merged = lora.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
