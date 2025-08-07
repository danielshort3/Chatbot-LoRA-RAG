import json
import os
import pathlib

import faiss
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from vgj_chat.config import CFG

MODEL_DIR = pathlib.Path("/opt/ml/model")
CACHE_DIR = pathlib.Path(os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if CFG.cuda and torch.cuda.is_available() else "cpu"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto" if DEVICE == "cuda" else {"": DEVICE},
)
INDEX = faiss.read_index(str(MODEL_DIR / "faiss.index"))
EMBEDDER = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE,
    cache_folder=str(CACHE_DIR),
)
METADATA = [json.loads(line) for line in open(MODEL_DIR / "meta.jsonl")]

app = FastAPI()


class Prompt(BaseModel):
    inputs: str


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/invocations")
def invoke(p: Prompt):
    query_emb = EMBEDDER.encode(p.inputs, normalize_embeddings=True)
    if query_emb.shape[0] != INDEX.d:
        raise ValueError(
            f"Embedding dimension {query_emb.shape[0]} does not match index dimension {INDEX.d}"
        )
    _, ids = INDEX.search(query_emb.reshape(1, -1), 5)
    hits = [METADATA[i] for i in ids[0]]
    context_parts = []
    sources: list[str] = []
    for h in hits:
        src = h.get("source") or h.get("url") or "unknown"
        text = h.get("text")
        if text:
            context_parts.append(f"<CONTEXT>{src}: {text}</CONTEXT>")
        if src not in sources:
            sources.append(src)
    context = "\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": "You are a friendly travel expert representing Visit Grand Junction.",
        },
        {"role": "user", "content": f"{context}\n\n{p.inputs}"},
    ]

    # --- tokenise prompt ---------------------------------------------------
    input_ids = TOKENIZER.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(MODEL.device)
    n_prompt = input_ids.shape[1]  # prompt token count

    # --- generate ----------------------------------------------------------
    output = MODEL.generate(
        input_ids=input_ids,
        max_new_tokens=CFG.max_new_tokens,
        temperature=0.2,  # keeps it concise
    )

    # --- token accounting --------------------------------------------------
    n_total = output.shape[1]  # prompt + continuation
    n_answer = n_total - n_prompt  # just the new tokens
    print(f"[TOKENS] prompt={n_prompt}  answer={n_answer}  total={n_total}")

    # (optional) include in JSON payload
    # ---------------------------------------------------------------
    DISCLAIMER = (
        "⚠️  Portfolio demo only. "
        "Opinions are Daniel Short’s and do **not** represent Visit Grand Junction "
        "or the City of Grand Junction.\n\n"
    )

    answer_text = TOKENIZER.decode(
        output[0][n_prompt:], skip_special_tokens=True
    ).strip()

    return {
        "generated_text": DISCLAIMER + answer_text,
        "sources": sources,
        "token_usage": {
            "prompt": n_prompt,
            "answer": n_answer,
            "total": n_total,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
