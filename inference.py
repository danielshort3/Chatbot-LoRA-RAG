import json
import os
from typing import Any, Dict

import faiss
import torch
from sagemaker_inference import model_server
from transformers import AutoModelForCausalLM, AutoTokenizer

from vgj_chat.config import CFG

SYSTEM_PROMPT = (
    "You are a friendly travel expert representing Visit Grand Junction. "
    "Use the supplied context excerpts to answer questions about Grand Junction, "
    "Colorado and its surroundings in a warm, adventurous tone that highlights "
    "outdoor recreation, local culture, and natural beauty. If the context does "
    "not contain the needed information, say you don't know and recommend "
    "checking official Visit Grand Junction resources."
)


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
    return {"lm": model, "tok": tok, "index": index, "meta": meta}


def predict_fn(data, ctx):
    mdl = ctx
    prompt = data["inputs"]
    top_k = data.get("top_k", 3)

    emb_query = (
        mdl["lm"]
        .get_input_embeddings()(
            torch.tensor(mdl["tok"](prompt)["input_ids"]).to("cuda")
        )
        .mean(0)
        .detach()
        .cpu()
        .numpy()
    )

    _distances, indices = mdl["index"].search(emb_query[None, :], top_k)
    retrieved = "\n".join(
        f"<CONTEXT>{mdl['meta'][idx]['text']}</CONTEXT>" for idx in indices[0]
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{retrieved}\n\n{prompt}"},
    ]
    aug_prompt = mdl["tok"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mdl["tok"](aug_prompt, return_tensors="pt").to("cuda")
    gen_ids = mdl["lm"].generate(**input_ids, max_new_tokens=CFG.max_new_tokens)
    answer = mdl["tok"].decode(gen_ids[0], skip_special_tokens=True)
    return {"generated_text": answer}


if __name__ == "__main__":
    model_server.start_model_server(handler_service=predict_fn, model_fn=model_fn)
