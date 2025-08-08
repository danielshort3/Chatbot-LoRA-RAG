"""Auto-generate Q&A pairs from crawled pages."""

from __future__ import annotations

import json
import os
import random
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
CTX_MAX = 5
SPECIAL_TOKENS = {"additional_special_tokens": ["<CONTEXT>", "</CONTEXT>"]}

TXT_DIR = Path("data/html_txt")
RAW_HTML_DIR = Path("data/raw_html")

# store auto-generated pairs under data/dataset/
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")
MODEL_CACHE = Path("data/model_cache")


def _gen_question(passage: str, tok: AutoTokenizer, llm: AutoModelForCausalLM) -> str:
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


def _gen_context(
    question: str, answer_part: str, tok: AutoTokenizer, llm: AutoModelForCausalLM
) -> str:
    """Generate a short context passage supporting *answer_part* for *question*.

    The language model samples a 2–3 sentence block that helps address the
    portion of the answer specified by ``answer_part``. Sampling is enabled so
    repeated calls yield varied passages.
    """

    sys = (
        "You are a helpful travel assistant. Invent a short context passage of "
        "2-3 sentences that could help answer the QUESTION, focusing on the "
        "ANSWER_SNIPPET. Do not answer the question directly."
    )
    prompt = (
        f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\nQUESTION: {question}\n"
        f"ANSWER_SNIPPET: {answer_part}\n[/INST]"
    )
    ids = tok(prompt, return_tensors="pt").to(llm.device)
    for _ in range(3):
        with torch.no_grad():
            out = llm.generate(
                **ids,
                max_new_tokens=80,
                pad_token_id=tok.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )[0]
        ctx = tok.decode(out[ids.input_ids.shape[-1] :], skip_special_tokens=True).strip()
        sentences = re.findall(r"[^.!?]+[.!?]", ctx)
        if len(sentences) >= 2:
            keep = min(len(sentences), random.choice([2, 3]))
            return " ".join(s.strip() for s in sentences[:keep])
    return ctx


def _choose_num_ctx(max_ctx: int) -> int:
    """Sample how many context snippets to include for a question.

    The count follows a normal distribution centered at 2.5 and is
    clamped to the inclusive range [0, 5] as well as the provided
    ``max_ctx`` limit.
    """

    num = round(random.gauss(2.5, 1))
    num = max(0, min(5, num))
    return min(num, max_ctx)


BOILER_PAT = re.compile(
    r"(click here|minute read|photo credit|browser is not supported)", re.I
)


def build_auto_dataset() -> None:
    if AUTO_QA_JL.exists():
        print(f"{AUTO_QA_JL} exists; skipping dataset build")
        return

    hf_token = os.getenv("VGJ_HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    tok = AutoTokenizer.from_pretrained(
        LLM_NAME, use_fast=True, token=hf_token, cache_dir=MODEL_CACHE
    )
    tok.add_special_tokens(SPECIAL_TOKENS)
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
            token=hf_token,
            cache_dir=MODEL_CACHE,
        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            torch_dtype=torch.float32,
            token=hf_token,
            cache_dir=MODEL_CACHE,
        )

    llm.resize_token_embeddings(len(tok))

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
        question = _gen_question(passage, tok, llm)
        words, answer_words, used_paras = 0, [], []
        for p in paras:
            if words + len(p.split()) > ANSWER_TOK_CAP:
                break
            answer_words.extend(p.split())
            words += len(p.split())
            used_paras.append(p)
        answer = " ".join(answer_words) or paras[0]
        if BOILER_PAT.search(answer):
            skipped += 1
            continue
        available_parts = used_paras
        ctx_blocks: list[str] = []
        if available_parts:
            max_ctx = min(len(available_parts), CTX_MAX)
            num_ctx = _choose_num_ctx(max_ctx)
            if num_ctx:
                ctx_parts = random.sample(available_parts, k=num_ctx)
                for part in ctx_parts:
                    ctx_blocks.append(
                        f"<CONTEXT>\n{_gen_context(question, part, tok, llm)}\n</CONTEXT>"
                    )
                random.shuffle(ctx_blocks)
        ctx_str = "\n\n".join(ctx_blocks)
        prompt = f"{ctx_str}\n\n{question}" if ctx_blocks else question
        auto_examples.append({"input": prompt, "output": answer})
    with AUTO_QA_JL.open("w") as f:
        for ex in auto_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {len(auto_examples):,} clean pairs → {AUTO_QA_JL}")
