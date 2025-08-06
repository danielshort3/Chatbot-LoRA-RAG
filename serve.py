import json
import pathlib

import faiss
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from vgj_chat.config import CFG
from transformers import StoppingCriteria, StoppingCriteriaList

MODEL_DIR = pathlib.Path("/opt/ml/model")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype="auto", device_map="auto"
)
INDEX = faiss.read_index(str(MODEL_DIR / "faiss.index"))
EMBEDDER = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_folder=str(MODEL_DIR),
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
        context_parts.append(f"{src}: {h['text']}")
        if src not in sources:
            sources.append(src)
    context = "\n".join(context_parts)
    
    PROMPT_TMPL = """<s>[INST] 
    You are a helpful Grand Junction travel assistant.  
    Answer using **only** the sources below.  
    Write **exactly two short paragraphs** (4-6 sentences total).  
    Do **not** repeat the question, do **not** add lists, FAQs, or headings.  
    When you are finished, output the word ###END on its own line – nothing after that. [/INST]

    {context}

    [INST] {question} [/INST]
    """
    prompt = PROMPT_TMPL.format(context=context, question=p.inputs)

    # --- tokenise prompt ---------------------------------------------------
    enc = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    n_prompt = enc.input_ids.shape[1]             # prompt token count


    class StopOnEnd(StoppingCriteria):
        END_IDS = TOKENIZER("###END").input_ids
        def __call__(self, input_ids, scores, **kwargs):
            # stop if the last |END_IDS| tokens equal the sentinel
            return list(input_ids[0][-len(self.END_IDS):].cpu().numpy()) == self.END_IDS

    stops = StoppingCriteriaList([StopOnEnd()])

    # --- generate ----------------------------------------------------------
    output = MODEL.generate(
        **enc, 
        max_new_tokens=CFG.max_new_tokens,
        stopping_criteria=stops,
        temperature=0.2,              # keeps it concise
    )

    # --- token accounting --------------------------------------------------
    n_total  = output.shape[1]                    # prompt + continuation
    n_answer = n_total - n_prompt                 # just the new tokens
    print(f"[TOKENS] prompt={n_prompt}  answer={n_answer}  total={n_total}")

    # (optional) include in JSON payload
    # ---------------------------------------------------------------
    DISCLAIMER = (
        "⚠️  Portfolio demo only. "
        "Opinions are Daniel Short’s and do **not** represent Visit Grand Junction "
        "or the City of Grand Junction.\n\n"
    )

    answer_text = TOKENIZER.decode(output[0][n_prompt:], skip_special_tokens=True)
    answer_text = answer_text.split("###END")[0].strip()

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
