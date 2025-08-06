from __future__ import annotations

import logging
import threading
from pathlib import Path

import faiss  # type: ignore
import torch  # type: ignore

try:  # pragma: no cover - dependency stubbed in tests
    from huggingface_hub import login
except Exception:  # pragma: no cover - fallback for tests

    def login(*_a, **_k) -> None:
        pass


from sentence_transformers import CrossEncoder, SentenceTransformer

# Import `pipeline` from modern transformers, fall back to legacy path if needed
try:
    from transformers import pipeline
except (ModuleNotFoundError):
    from transformers.pipeline import pipeline

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from ...config import CFG
from ...data.io import load_index, load_metadata

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
_STATE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# resource initialisation
# ---------------------------------------------------------------------------


def _boot() -> (
    tuple[
        faiss.Index,
        list[str],
        list[str],
        SentenceTransformer,
        CrossEncoder,
        pipeline,
    ]
):
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
    reranker = CrossEncoder(CFG.rerank_model, device=device, cache_dir=MODEL_CACHE)

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
_RETRIEVAL_DISABLED = False


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


__all__ = [
    "CFG",
    "CHAT",
    "INDEX",
    "TEXTS",
    "URLS",
    "EMBEDDER",
    "RERANKER",
    "BASELINE_CHAT",
    "_STATE_LOCK",
    "_BOOTED",
    "_RETRIEVAL_DISABLED",
    "_ensure_boot",
    "logger",
    "MODEL_CACHE",
]
