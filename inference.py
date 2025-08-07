import json
import os
from pathlib import Path
from typing import Any, Dict

import faiss
import torch
from sagemaker_inference import model_server
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from vgj_chat.config import CFG

CACHE_DIR = Path(os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def model_fn(model_dir: str) -> Dict[str, Any]:
    device = "cuda" if CFG.cuda and torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else {"": device},
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    index = faiss.read_index(os.path.join(model_dir, "faiss.index"))
    meta_path = os.path.join(model_dir, "meta.jsonl")
    meta = [json.loads(line) for line in open(meta_path)]
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
        cache_folder=str(CACHE_DIR),
    )
    return {
        "lm": model,
        "tok": tok,
        "index": index,
        "meta": meta,
        "encoder": embedder,
        "device": device,
    }


def predict_fn(data, ctx):
    mdl = ctx
    prompt = data["inputs"]
    top_k = data.get("top_k", 3)

    emb_query = mdl["encoder"].encode(prompt, normalize_embeddings=True)
    if emb_query.shape[0] != mdl["index"].d:
        raise ValueError(
            f"Embedding dimension {emb_query.shape[0]} does not match index dimension {mdl['index'].d}"
        )
    _distances, indices = mdl["index"].search(emb_query.reshape(1, -1), top_k)
    context_parts = []
    sources: list[str] = []
    for idx in indices[0]:
        meta = mdl["meta"][idx]
        src = meta.get("source") or meta.get("url") or "unknown"
        text = meta.get("text")
        if text:
            context_parts.append(f"<CONTEXT>{src}: {text}</CONTEXT>")
        if src not in sources:
            sources.append(src)
    context = "\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly travel expert representing Visit Grand Junction. "
                "Only discuss attractions in Grand Junction, Colorado. "
                "If asked about other destinations, prices, or deals, politely state that you can only "
                "talk about Grand Junction. Limit your response to one or two short paragraphs."
            ),
        },
        {"role": "user", "content": f"{context}\n\n{prompt}"},
    ]
    device = mdl["device"]
    input_ids = (
        mdl["tok"]
        .apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        .to(device)
    )
    n_prompt = input_ids.shape[1]
    gen_ids = mdl["lm"].generate(
        input_ids=input_ids,
        max_new_tokens=CFG.max_new_tokens,
        temperature=0.2,
        eos_token_id=mdl["tok"].eos_token_id,
        pad_token_id=mdl["tok"].eos_token_id,
    )
    n_total = gen_ids.shape[1]
    n_answer = n_total - n_prompt
    answer_text = mdl["tok"].decode(
        gen_ids[0][n_prompt:], skip_special_tokens=True
    ).strip()

    DISCLAIMER = (
        "⚠️  Portfolio demo only. "
        "Opinions are Daniel Short’s and do **not** represent Visit Grand Junction "
        "or the City of Grand Junction.\n\n"
    )

    src_text = ", ".join(sources)
    generated = DISCLAIMER + answer_text
    if src_text:
        generated += f"\n\nSources: {src_text}"

    return {
        "generated_text": generated,
        "sources": sources,
        "token_usage": {
            "prompt": n_prompt,
            "answer": n_answer,
            "total": n_total,
        },
    }


if __name__ == "__main__":
    model_server.start_model_server(handler_service=predict_fn, model_fn=model_fn)
