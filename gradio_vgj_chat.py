#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────────────────────
#  gradio_vgj_chat.py
#  Unofficial Visit Grand Junction demo chatbot (RAG + LoRA)
#  Debug to console; boiler‑plate filters; similarity guard‑rail.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
import re
import threading  # NEW
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore
import gradio as gr
import numpy as np  # type: ignore
import torch  # type: ignore
from peft import PeftModel  # type: ignore
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)

from vgj_chat.data.io import load_index


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    # paths
    index_path: Path = Path("faiss.index")
    meta_path: Path = Path("meta.jsonl")
    lora_dir: Path = Path("lora-vgj-checkpoint")

    # models
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "BAAI/bge-reranker-base"

    # RAG settings
    top_k: int = 5
    score_min: float = 0.0
    max_new_tokens: int = 512

    # similarity guard‑rail
    sim_threshold: float = 0.80

    # misc
    cuda: bool = torch.cuda.is_available()
    debug: bool = False


CFG = Config()

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.DEBUG if CFG.debug else logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vgj_chat")

# ──────────────────────────────────────────────────────────────────────────────
#  Boiler‑plate & FAQ stripping helpers
# ──────────────────────────────────────────────────────────────────────────────
FAQ_RX = re.compile(r"^[QA]:", re.I)
FOOTER_RX = re.compile(
    r"(visit\s+grand\s+junction\s+is|©|all\s+rights\s+reserved|privacy\s+policy)",
    re.I,
)


def _clean(text: str) -> str:
    """
    Remove FAQ markers and obvious boiler‑plate / footer lines
    so the bot can’t quote long blocks verbatim.
    """
    cleaned_lines = []
    for line in text.splitlines():
        if FAQ_RX.match(line.strip()):
            continue
        if FOOTER_RX.search(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


# ──────────────────────────────────────────────────────────────────────────────
#  Resource initialisation
# ──────────────────────────────────────────────────────────────────────────────
def _boot() -> tuple[
    faiss.Index,
    list[str],
    list[str],
    SentenceTransformer,
    CrossEncoder,
    pipeline,
]:
    logger.info("Loading FAISS index and metadata …")
    index: faiss.Index = load_index(CFG.index_path)
    raw_meta = [json.loads(l) for l in CFG.meta_path.read_text().splitlines()]
    texts = [_clean(m["text"]) for m in raw_meta]  # ← pre‑filter once
    urls = [m["url"] for m in raw_meta]

    device = "cuda" if CFG.cuda else "cpu"
    logger.info(
        "Using device: %s (CUDA available: %s)", device, torch.cuda.is_available()
    )
    logger.info("Initialising embedding & re‑rank models …")
    embedder = SentenceTransformer(CFG.embed_model, device=device)
    reranker = CrossEncoder(CFG.rerank_model, device=device)

    logger.info("Loading LoRA‑merged generator …")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    lora = PeftModel.from_pretrained(base, str(CFG.lora_dir))
    merged = lora.merge_and_unload()

    chat_pipe = pipeline(
        "text-generation",
        model=merged,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=CFG.max_new_tokens,
        do_sample=False,
    )
    logger.info("Boot complete.")
    return index, texts, urls, embedder, reranker, chat_pipe


INDEX, TEXTS, URLS, EMBEDDER, RERANKER, CHAT = _boot()


# ──────────────────────────────────────────────────────────────────────────────
#  Retrieval
# ──────────────────────────────────────────────────────────────────────────────
def _retrieve_unique(query: str) -> List[Tuple[float, str, str]]:
    """Return the top-K unique passages for *query* sorted by score."""

    logger.debug("🔍 Query: %s", query)

    q_vec = EMBEDDER.encode(query, normalize_embeddings=True).astype("float32")[None, :]
    _d, idx = INDEX.search(q_vec, 100)

    # apply cleaning again (belt‑and‑braces)
    candidates = [(_clean(TEXTS[i]), URLS[i]) for i in idx[0]]
    raw_scores = RERANKER.predict([(query, t) for t, _ in candidates])

    # deduplicate by URL on max score
    best: dict[str, Tuple[float, str]] = {}
    for score, (text, url) in zip(raw_scores, candidates):
        if score < CFG.score_min:
            continue
        best[url] = max(best.get(url, (0, "")), (score, text))

    uniques = sorted(
        ((s, t, u) for u, (s, t) in best.items()),
        key=lambda x: x[0],
        reverse=True,
    )[: CFG.top_k]

    logger.debug("Retrieved %d unique passages.", len(uniques))
    return uniques


# ──────────────────────────────────────────────────────────────────────────────
#  Similarity guard‑rail
# ──────────────────────────────────────────────────────────────────────────────
def _too_similar(answer: str, passages: List[Tuple[float, str, str]]) -> bool:
    """
    Returns True if the generated answer overlaps too closely
    with any retrieved passage (cosine ≥ threshold).
    """
    ans_vec = EMBEDDER.encode(answer, normalize_embeddings=True).astype("float32")
    for score, text, _url in passages:
        src_vec = EMBEDDER.encode(text[:512], normalize_embeddings=True).astype(
            "float32"
        )  # small slice
        cos_sim = float(np.dot(ans_vec, src_vec))  # because both are unit vectors
        if cos_sim >= CFG.sim_threshold:
            logger.warning(
                "Similarity %.2f ≥ %.2f – refusing/paraphrasing.",
                cos_sim,
                CFG.sim_threshold,
            )
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  Chat orchestration
# ──────────────────────────────────────────────────────────────────────────────
def _answer_stream(history: list[dict[str, str]]):
    """
    Stream a RAG-grounded answer into Gradio.  Yields (chatbox_history, state)
    each time a new token arrives so the UI updates live.
    """
    user_q = history[-1]["content"]

    # ---------------- retrieval ----------------
    passages = _retrieve_unique(user_q)
    if not passages:
        history.append(
            {
                "role": "assistant",
                "content": "Sorry, I couldn’t find anything relevant.",
            }
        )
        yield history, history
        return

    # ---------------- RAG prompt ----------------
    src_block = "\n\n".join(
        f"[{i+1}] {url}\n{text}" for i, (_s, text, url) in enumerate(passages)
    )
    prompt = (
        "Answer the *single* question below using only the listed sources. "
        "Do not add additional questions, FAQs, or headings. "
        "Cite each fact like [1].\n\n"
        f"{src_block}\n\nQ: {user_q}\nA:"
    )

    # ---------------- streaming generation ----------------
    tok = CHAT.tokenizer
    model = CHAT.model
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if "attention_mask" not in inputs:  # safety for some HF builds
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    threading.Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=CFG.max_new_tokens,
            do_sample=False,
            streamer=streamer,
        ),
        daemon=True,
    ).start()

    # add an empty assistant message that we'll fill incrementally
    history.append({"role": "assistant", "content": ""})
    partial = ""
    for token in streamer:
        partial += token
        history[-1]["content"] = partial
        yield history, history  # live update

    final_answer = history[-1]["content"].strip()

    # ---------------- append sources ----------------
    sources_md = "\n".join(f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages))
    history[-1]["content"] = f"{final_answer}\n\n**Sources**\n{sources_md}"
    yield history, history  # final, complete message


def _user_submit(msg: str, hist: List[dict[str, str]]) -> tuple[str, list]:
    hist.append({"role": "user", "content": msg})
    logger.debug("━ User question: %s", msg)
    return "", hist


# ──────────────────────────────────────────────────────────────────────────────
#  Gradio UI
# ──────────────────────────────────────────────────────────────────────────────
page_title = (
    "Unofficial Visit Grand Junction Demo – not endorsed by VGJ"  # <title> element
)

with gr.Blocks(theme=gr.themes.Soft(), title=page_title) as demo:
    gr.Markdown(
        (
            "## 💬 Unofficial Visit Grand Junction Demo Chatbot\n"
            "<small>Portfolio prototype, **not** endorsed by Visit Grand Junction. "
            "Content sourced from public VGJ blogs under a fair‑use rationale.</small>"
        )
    )

    chat_state = gr.State([])

    chatbox = gr.Chatbot(height=450, type="messages", label="Conversation")
    textbox = gr.Textbox(
        placeholder="Ask about Grand Junction…",
        show_label=False,
        container=False,
    )

    textbox.submit(
        _user_submit,
        inputs=[textbox, chat_state],
        outputs=[textbox, chat_state],
    ).then(
        _answer_stream,  # generator streams tokens
        inputs=[chat_state],
        outputs=[chatbox, chat_state],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
