import json
import pathlib

import faiss
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODEL_DIR = pathlib.Path("/opt/ml/model")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype="auto", device_map="auto"
)
INDEX = faiss.read_index(str(MODEL_DIR / "faiss.index"))
EMBEDDER = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
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
    context = " ".join(METADATA[i]["text"] for i in ids[0])
    prompt = (
        "Answer the *single* question below using only the listed sources. "
        "Do not add additional questions, FAQs, or headings. "
        "Begin your answer with 'This portfolio project was created by Daniel Short. "
        "Views expressed do not represent Visit Grand Junction or the City of Grand Junction.' "
        "Limit your answer to one or two short paragraphs.\n\n"
        f"{context}\nUser: {p.inputs}\nAssistant:"
    )
    output = MODEL.generate(
        **TOKENIZER(prompt, return_tensors="pt").to(MODEL.device),
        max_new_tokens=200,
    )
    answer = TOKENIZER.decode(output[0], skip_special_tokens=True).split("Assistant:")[
        -1
    ]
    return {"generated_text": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
