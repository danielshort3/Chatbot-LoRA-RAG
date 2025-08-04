"""Auto-generate Q&A pairs from crawled pages."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
import trafilatura
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
PARA_MAX = 3
ANSWER_TOK_CAP = 220

TXT_DIR = Path("data/html_txt")
RAW_HTML_DIR = Path("data/raw_html")

# store auto-generated pairs under data/dataset/
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")
MODEL_CACHE = Path("data/model_cache")

# authenticate for gated base model if token available
HF_TOKEN = os.getenv("VGJ_HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

tok = AutoTokenizer.from_pretrained(
    LLM_NAME, use_fast=True, token=HF_TOKEN, cache_dir=MODEL_CACHE
)
if torch.cuda.is_available():
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=HF_TOKEN,
        cache_dir=MODEL_CACHE,
    )
else:
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float32,
        token=HF_TOKEN,
        cache_dir=MODEL_CACHE,
    )


def gen_question(passage: str) -> str:
    sys = (
        "You are a helpful travel assistant. Read the PASSAGE and invent one "
        "concise, natural-sounding traveler question that could be answered "
        "by the same passage. Return ONLY the question text."
    )
    prompt = f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\nPASSAGE:\n'''{passage}'''\n[/INST]"
    ids = tok(prompt, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**ids, max_new_tokens=40, pad_token_id=tok.eos_token_id)[0]
    q = tok.decode(out[ids.input_ids.shape[-1] :], skip_special_tokens=True).strip()
    return q if q.endswith("?") else q + "?"


BOILER_PAT = re.compile(
    r"(click here|minute read|photo credit|browser is not supported)", re.I
)


def build_auto_dataset() -> None:
    if AUTO_QA_JL.exists():
        print(f"{AUTO_QA_JL} exists; skipping dataset build")
        return
    AUTO_QA_JL.parent.mkdir(parents=True, exist_ok=True)
    auto_examples = []
    skipped = 0
    for txt_f in tqdm(sorted(TXT_DIR.glob("*.txt")), desc="auto-QA", unit="page"):
        html = (RAW_HTML_DIR / f"{txt_f.stem}.html").read_text()
        text = trafilatura.extract(html) or ""
        paras = [p.strip() for p in text.splitlines() if len(p.split()) > 25][:PARA_MAX]
        if not paras:
            continue
        passage = "\n\n".join(paras)
        question = gen_question(passage)
        words, answer_words = 0, []
        for p in paras:
            if words + len(p.split()) > ANSWER_TOK_CAP:
                break
            answer_words.extend(p.split())
            words += len(p.split())
        answer = " ".join(answer_words) or paras[0]
        if BOILER_PAT.search(answer):
            skipped += 1
            continue
        auto_examples.append({"input": question, "output": answer})
    with AUTO_QA_JL.open("w") as f:
        for ex in auto_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {len(auto_examples):,} clean pairs → {AUTO_QA_JL}")
