from __future__ import annotations

import logging
import threading
from typing import Generator, List, Tuple

import faiss  # type: ignore
import torch  # type: ignore
from huggingface_hub import login
from peft import PeftModel  # type: ignore
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)

from ..config import CFG
from ..data.io import load_index, load_metadata
from .guards import too_similar

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.DEBUG if CFG.debug else logging.INFO,
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# resource initialisation
# ---------------------------------------------------------------------------


def _boot() -> tuple[
    faiss.Index,
    list[str],
    list[str],
    SentenceTransformer,
    CrossEncoder,
    pipeline,
]:
    if CFG.hf_token is not None:
        login(token=CFG.hf_token)
    logger.info("Loading FAISS index and metadata …")
    index = load_index(CFG.index_path)
    texts, urls = load_metadata(CFG.meta_path)

    device = "cuda" if CFG.cuda else "cpu"
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
    lora = PeftModel.from_pretrained(base, CFG.lora_dir)
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


INDEX: faiss.Index | None = None
TEXTS: list[str] | None = None
URLS: list[str] | None = None
EMBEDDER: SentenceTransformer | None = None
RERANKER: CrossEncoder | None = None
CHAT: pipeline | None = None
_BOOTED = False


def _ensure_boot() -> None:
    """Boot the heavy resources on demand."""
    global INDEX, TEXTS, URLS, EMBEDDER, RERANKER, CHAT, _BOOTED
    if not _BOOTED:
        (
            INDEX,
            TEXTS,
            URLS,
            EMBEDDER,
            RERANKER,
            CHAT,
        ) = _boot()
        _BOOTED = True


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def retrieve_unique(query: str) -> List[Tuple[float, str, str]]:
    """Return the top-K unique passages for *query* sorted by score."""

    _ensure_boot()
    assert EMBEDDER and INDEX and TEXTS and URLS and RERANKER

    logger.debug("🔍 Query: %s", query)

    q_vec = EMBEDDER.encode(query, normalize_embeddings=True).astype("float32")[None, :]
    _d, idx = INDEX.search(q_vec, 100)

    candidates = [(TEXTS[i], URLS[i]) for i in idx[0]]
    raw_scores = RERANKER.predict([(query, t) for t, _ in candidates])

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


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def answer_stream(
    history: List[dict[str, str]],
) -> Generator[Tuple[List[dict[str, str]], List[dict[str, str]]], None, None]:
    """Stream a RAG-grounded answer for the Gradio UI."""
    _ensure_boot()
    assert CHAT and EMBEDDER
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
        "Cite each fact like [1].\n\n"
        f"{src_block}\n\nQ: {user_q}\nA:"
    )

    tok = CHAT.tokenizer
    model = CHAT.model
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if "attention_mask" not in inputs:
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

    history.append({"role": "assistant", "content": ""})
    partial = ""
    for token in streamer:
        partial += token
        history[-1]["content"] = partial
        yield history, history

    final_answer = history[-1]["content"].strip()

    if too_similar(final_answer, passages, EMBEDDER, CFG.sim_threshold):
        final_answer = "Sorry, the answer is too similar to the source material."

    sources_md = "\n".join(f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages))
    history[-1]["content"] = f"{final_answer}\n\n**Sources**\n{sources_md}"
    yield history, history


def chat(question: str) -> str:
    """Return a single answer string to *question*."""
    _ensure_boot()
    assert CHAT and EMBEDDER

    passages = retrieve_unique(question)
    if not passages:
        return "Sorry, I couldn’t find anything relevant."

    src_block = "\n\n".join(
        f"[{i+1}] {url}\n{text}" for i, (_s, text, url) in enumerate(passages)
    )
    prompt = (
        "Answer the *single* question below using only the listed sources. "
        "Do not add additional questions, FAQs, or headings. "
        "Cite each fact like [1].\n\n"
        f"{src_block}\n\nQ: {question}\nA:"
    )

    generated = CHAT(prompt)[0]["generated_text"]
    answer = generated[len(prompt) :].strip()

    if too_similar(answer, passages, EMBEDDER, CFG.sim_threshold):
        answer = "Sorry, the answer is too similar to the source material."

    sources_md = "\n".join(f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages))
    return f"{answer}\n\n**Sources**\n{sources_md}"
