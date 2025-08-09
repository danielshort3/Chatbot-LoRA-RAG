from __future__ import annotations

from typing import List, Tuple

import torch  # type: ignore

from . import boot as _boot
from .retrieval import retrieve_unique

system_prompt = """You are a friendly travel expert representing Visit Grand Junction. 
Answer questions about Grand Junction, Colorado and its surroundings in a warm, adventurous tone that highlights outdoor recreation, local culture, and natural beauty. 
Keep responses concise, factual, and helpful; avoid speculation or invented details. 
If you are unsure or lack information, say so and suggest checking official Visit Grand Junction resources."""


def _build_messages(question: str, passages: List[Tuple[float, str, str]]):
    ctx_blocks = [f"<CONTEXT>\n{text.strip()}\n</CONTEXT>" for _, text, _ in passages]
    user = ("\n\n".join(ctx_blocks) + "\n\n" if ctx_blocks else "") + question.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    sources = [url for _, _, url in passages]
    return messages, sources


def chat(question: str):
    """Return an answer string and accompanying sources."""
    _boot._ensure_boot()
    assert _boot.CHAT and _boot.EMBEDDER

    with _boot._STATE_LOCK:
        passages = retrieve_unique(question)
        if not passages:
            return {"answer": "Sorry, I couldn’t find anything relevant.", "sources": []}

        messages, sources = _build_messages(question, passages)

        tok = _boot.CHAT.tokenizer
        model = _boot.CHAT.model

        inputs = tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        n_prompt = inputs.shape[1]
        max_new = min(_boot.CFG.max_new_tokens, model.config.max_position_embeddings - n_prompt - 32)

        generated = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new,
            do_sample=False,
        )

        answer = tok.decode(
            generated[0][n_prompt:], skip_special_tokens=True
        ).strip()

    return {"answer": answer, "sources": sources}


__all__ = ["chat"]
