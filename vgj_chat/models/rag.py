from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Tuple

import faiss  # type: ignore
import torch  # type: ignore

try:
    from huggingface_hub import login
except Exception:  # pragma: no cover - fallback for tests

    def login(*_a, **_k) -> None:
        pass


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

logger = logging.getLogger(__name__)
MODEL_CACHE = Path("data/model_cache")


def _configure_logging() -> None:
    """Configure logging based on :data:`CFG.debug`."""
    level = logging.DEBUG if CFG.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )
    logger.setLevel(level)


# Configure global logging and state lock
_configure_logging()

# Lock to guard global model state when switching pipelines
_STATE_LOCK = threading.Lock()

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
    logger.info(
        "Using device: %s (CUDA available: %s)", device, torch.cuda.is_available()
    )
    logger.info("Initialising embedding & re‑rank models …")
    embedder = SentenceTransformer(
        CFG.embed_model, device=device, cache_folder=str(MODEL_CACHE)
    )
    reranker = CrossEncoder(
        CFG.rerank_model, device=device, cache_dir=MODEL_CACHE
    )

    logger.info("Loading quantised model …")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    if not CFG.merged_model_dir.exists():
        raise FileNotFoundError(
            f"Expected merged model directory at {CFG.merged_model_dir}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(CFG.merged_model_dir), use_fast=True)
    merged = AutoModelForCausalLM.from_pretrained(
        str(CFG.merged_model_dir),
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )

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
BASELINE_CHAT: pipeline | None = None
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


def _load_baseline_chat() -> pipeline:
    """Load the base model without LoRA for baseline comparisons."""
    logger.info("Loading baseline generator …")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(
        CFG.base_model, use_fast=True, cache_dir=MODEL_CACHE
    )
    model = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        max_new_tokens=CFG.max_new_tokens,
        do_sample=True,
    )


_RETRIEVAL_DISABLED = False


@contextmanager
def _baseline_mode() -> Generator[None, None, None]:
    """Temporarily disable retrieval and LoRA for a baseline generation."""

    global CHAT, INDEX, TEXTS, URLS, EMBEDDER, RERANKER, BASELINE_CHAT, _RETRIEVAL_DISABLED

    _STATE_LOCK.acquire()
    try:
        if BASELINE_CHAT is None:
            BASELINE_CHAT = _load_baseline_chat()

        old_values = (
            CHAT,
            INDEX,
            TEXTS,
            URLS,
            EMBEDDER,
            RERANKER,
            _RETRIEVAL_DISABLED,
        )
        CHAT = BASELINE_CHAT
        INDEX = TEXTS = URLS = EMBEDDER = RERANKER = None
        _RETRIEVAL_DISABLED = True

        yield
    finally:
        (
            CHAT,
            INDEX,
            TEXTS,
            URLS,
            EMBEDDER,
            RERANKER,
            _RETRIEVAL_DISABLED,
        ) = old_values
        _STATE_LOCK.release()


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def retrieve_unique(query: str) -> List[Tuple[float, str, str]]:
    """Return the top-K unique passages for *query* sorted by score."""

    if _RETRIEVAL_DISABLED:
        return []

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
        "Begin your answer with 'This portfolio project was created by Daniel Short. "
        "Views expressed do not represent Visit Grand Junction or the City of Grand Junction.' "
        "Limit your answer to one or two short paragraphs. "
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

    sources_md = "\n".join(f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages))
    history[-1]["content"] = f"{final_answer}\n\n**Sources**\n{sources_md}"
    yield history, history


def chat(question: str) -> str:
    """Return a single answer string to *question*."""
    _ensure_boot()
    assert CHAT and EMBEDDER

    with _STATE_LOCK:
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

        generated = CHAT(prompt)[0]["generated_text"]
        answer = generated[len(prompt) :].strip()

    sources_md = "\n".join(f"[{i+1}] {url}" for i, (_s, _t, url) in enumerate(passages))
    return f"{answer}\n\n**Sources**\n{sources_md}"


def run_enhanced(question: str) -> str:
    """Return an answer using RAG with LoRA and citations."""
    return chat(question)


def run_baseline(question: str) -> str:
    """Generate an answer without retrieval or LoRA adapters."""
    with _baseline_mode():
        assert CHAT
        generated = CHAT(
            question,
            do_sample=True,
            temperature=0.8,
            max_new_tokens=CFG.max_new_tokens,
        )[0]["generated_text"]
    return generated.strip()
