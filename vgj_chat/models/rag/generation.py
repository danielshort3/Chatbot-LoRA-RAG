from __future__ import annotations

import threading
from typing import Generator, List, Tuple

import torch  # type: ignore
from transformers import TextIteratorStreamer

from . import boot as _boot
from .boot import logger
from .retrieval import retrieve_unique


def answer_stream(
    history: List[dict[str, str]],
) -> Generator[Tuple[List[dict[str, str]], List[dict[str, str]]], None, None]:
    """Stream a RAG-grounded answer for the Gradio UI."""
    _boot._ensure_boot()
    assert _boot.CHAT and _boot.EMBEDDER
    user_q = history[-1]["content"]

    passages = retrieve_unique(user_q)
    if not passages:
        history.append(
            {
                "role": "assistant",
                "content": "Sorry, I couldn’t find anything relevant.",
            }
        )
        yield history, history
        return

    src_block = "\n\n".join(
        f"[{i+1}] {url}\n{text}" for i, (_s, text, url) in enumerate(passages)
    )
    prompt = (
        "Answer the *single* question below using only the listed sources. "
        "Do not add additional questions, FAQs, or headings. "
        "Begin your answer with 'This portfolio project was created by Daniel Short. "
        "Views expressed do not represent Visit Grand Junction or the City of Grand Junction.' "
        "Limit your answer to one or two short paragraphs. "
        "Cite each fact like [1].\n\n"
        f"{src_block}\n\nQ: {user_q}\nA:"
    )

    tok = _boot.CHAT.tokenizer
    model = _boot.CHAT.model
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    threading.Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=_boot.CFG.max_new_tokens,
            do_sample=False,
            streamer=streamer,
        ),
        daemon=True,
    ).start()

    history.append({"role": "assistant", "content": ""})
    partial = ""
    for token in streamer:
        partial += token
        history[-1]["content"] = partial
        yield history, history

    final_answer = history[-1]["content"].strip()

    sources_md = "\n".join(f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages))
    history[-1]["content"] = f"{final_answer}\n\n**Sources**\n{sources_md}"
    yield history, history


def chat(question: str) -> str:
    """Return a single answer string to *question*."""
    _boot._ensure_boot()
    assert _boot.CHAT and _boot.EMBEDDER

    with _boot._STATE_LOCK:
        passages = retrieve_unique(question)
        if not passages:
            return "Sorry, I couldn’t find anything relevant."

        src_block = "\n\n".join(
            f"[{i+1}] {url}\n{text}" for i, (_s, text, url) in enumerate(passages)
        )
        prompt = (
            "Answer the *single* question below using only the listed sources. "
            "Do not add additional questions, FAQs, or headings. "
            "Begin your answer with 'This portfolio project was created by Daniel Short. "
            "Views expressed do not represent Visit Grand Junction or the City of Grand Junction.' "
            "Limit your answer to one or two short paragraphs. "
            "Cite each fact like [1].\n\n"
            f"{src_block}\n\nQ: {question}\nA:"
        )

        generated = _boot.CHAT(prompt)[0]["generated_text"]
        answer = generated[len(prompt) :].strip()

    sources_md = "\n".join(f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages))
    return f"{answer}\n\n**Sources**\n{sources_md}"


def run_enhanced(question: str) -> str:
    """Return an answer using RAG with LoRA and citations."""
    return chat(question)


__all__ = ["answer_stream", "chat", "run_enhanced"]
