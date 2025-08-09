import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import faiss
import torch
from sagemaker_inference import model_server
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vgj_chat.config import CFG
from vgj_chat.utils.text import strip_metadata

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


system_prompt = """You are a friendly travel expert representing Visit Grand Junction. 
Answer questions about Grand Junction, Colorado and its surroundings in a warm, adventurous tone that highlights outdoor recreation, local culture, and natural beauty. 
Keep responses concise, factual, and helpful; avoid speculation or invented details. 
If you are unsure or lack information, say so and suggest checking official Visit Grand Junction resources."""


def predict_fn(data, ctx):
    mdl = ctx
    prompt = data["inputs"].strip()
    top_k = data.get("top_k", CFG.top_k)

    # --- retrieve ---
    emb_query = mdl["encoder"].encode(prompt, normalize_embeddings=True)
    if emb_query.shape[0] != mdl["index"].d:
        raise ValueError(
            f"Embedding dimension {emb_query.shape[0]} does not match index dimension {mdl['index'].d}"
        )

    _distances, indices = mdl["index"].search(emb_query.reshape(1, -1), top_k)

    # --- format context EXACTLY like training ---
    ctx_blocks, sources = [], []
    for idx in indices[0]:
        meta = mdl["meta"][idx]
        src = meta.get("source") or meta.get("url") or "unknown"
        text = (meta.get("text") or "").strip()
        if text:
            ctx_blocks.append(f"<CONTEXT>\n{text}\n</CONTEXT>")
        if src and src not in sources:
            sources.append(src)

    user_msg = ("\n\n".join(ctx_blocks) + "\n\n" if ctx_blocks else "") + prompt

    # --- messages: NO system message (training had none) ---
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    # --- tokenize & generate with a small safety margin ---
    device = mdl["device"]
    tok = mdl["tok"]
    lm = mdl["lm"]

    input_ids = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(device)

    n_prompt = input_ids.shape[1]
    ctx_limit = getattr(lm.config, "max_position_embeddings", tok.model_max_length)
    max_new = max(1, min(CFG.max_new_tokens, ctx_limit - n_prompt - 32))  # margin

    gen_ids = lm.generate(
        input_ids=input_ids,
        max_new_tokens=max_new,
        do_sample=False,  # keeps style concise/consistent
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    n_total = gen_ids.shape[1]
    n_answer = n_total - n_prompt
    answer_text = strip_metadata(
        tok.decode(gen_ids[0][n_prompt:], skip_special_tokens=True).strip()
    )

    return {
        "generated_text": answer_text,
        "sources": sources,
        "token_usage": {"prompt": n_prompt, "answer": n_answer, "total": n_total},
    }


if __name__ == "__main__":
    model_server.start_model_server(handler_service=predict_fn, model_fn=model_fn)
