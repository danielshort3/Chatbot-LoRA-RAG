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
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    index = faiss.read_index(os.path.join(model_dir, "faiss.index"))
    meta_path = os.path.join(model_dir, "meta.jsonl")
    meta = [json.loads(line) for line in open(meta_path)]
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=str(CACHE_DIR),
    )
    return {
        "lm": model,
        "tok": tok,
        "index": index,
        "meta": meta,
        "encoder": embedder,
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
    retrieved = "\n".join(
        f"<CONTEXT>{mdl['meta'][idx]['text']}</CONTEXT>" for idx in indices[0]
    )

    messages = [
        {
            "role": "system",
            "content": "You are a friendly travel expert representing Visit Grand Junction.",
        },
        {"role": "user", "content": f"{retrieved}\n\n{prompt}"},
    ]
    input_ids = (
        mdl["tok"]
        .apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        .to("cuda")
    )
    gen_ids = mdl["lm"].generate(input_ids=input_ids, max_new_tokens=CFG.max_new_tokens)
    answer = mdl["tok"].decode(
        gen_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    return {"generated_text": answer}


if __name__ == "__main__":
    model_server.start_model_server(handler_service=predict_fn, model_fn=model_fn)
