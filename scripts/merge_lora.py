import argparse
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import login

DEFAULT_BASE = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_LORA = "lora-vgj-checkpoint"
DEFAULT_OUT = "mistral-merged-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--lora-dir", default=DEFAULT_LORA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    token = os.getenv("VGJ_HF_TOKEN")
    if token:
        login(token=token)

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=qconfig,
        token=token,
    )
    lora = PeftModel.from_pretrained(base, args.lora_dir)
    merged = lora.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=token)
    merged.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Merged model saved to {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
