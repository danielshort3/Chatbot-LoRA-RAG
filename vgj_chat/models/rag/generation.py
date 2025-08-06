from __future__ import annotations

from typing import List, Tuple

import torch  # type: ignore

from . import boot as _boot
from .boot import logger
from .retrieval import retrieve_unique


# ───────────────────────────────────
# 🔧 helper: build the shared prompt
# ───────────────────────────────────
def _make_prompt(question: str,
                 passages: List[Tuple[float, str, str]]) -> str:
    src_block = "\n\n".join(
        f"[{i+1}] {url}\n{text}"
        for i, (_score, text, url) in enumerate(passages)
    )
    return (
        "Answer the *single* question below using only the listed sources. "
        "Do not add additional questions, FAQs, or headings. "
        "Begin your answer with 'This portfolio project was created by Daniel Short. "
        "Views expressed do not represent Visit Grand Junction or the City of Grand Junction.' "
        "Limit your answer to one or two short paragraphs. "
        "Cite each fact like [1].\n\n"
        f"{src_block}\n\nQ: {question}\nA:"
    )


# ───────────────────────────────────
# 🌐 public API
# ───────────────────────────────────
def chat(question: str) -> str:
    """Return a fully-formed, citation-rich answer string."""
    _boot._ensure_boot()
    assert _boot.CHAT and _boot.EMBEDDER

    with _boot._STATE_LOCK:
        passages = retrieve_unique(question)
        if not passages:
            return "Sorry, I couldn’t find anything relevant."

        prompt = _make_prompt(question, passages)

        tok = _boot.CHAT.tokenizer
        model = _boot.CHAT.model

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        generated = model.generate(
            **inputs,
            max_new_tokens=_boot.CFG.max_new_tokens,
            do_sample=False,
        )

        answer = tok.decode(generated[0], skip_special_tokens=True)[len(prompt):].strip()

    sources_md = "\n".join(
        f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages)
    )
    return f"{answer}\n\n**Sources**\n{sources_md}"


# backwards-compat alias; drop if unused
run_enhanced = chat


def answer_stream(question: str):
    """Yield a single answer for compatibility with older APIs."""
    yield chat(question)

__all__ = ["chat", "run_enhanced", "answer_stream"]
