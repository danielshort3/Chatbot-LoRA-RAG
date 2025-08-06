from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch  # type: ignore
from transformers.pipeline import pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from . import boot as _boot
from .boot import logger


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
        _boot.CFG.base_model, use_fast=True, cache_dir=_boot.MODEL_CACHE
    )
    model = AutoModelForCausalLM.from_pretrained(
        _boot.CFG.base_model,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=_boot.MODEL_CACHE,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        max_new_tokens=_boot.CFG.max_new_tokens,
        do_sample=True,
    )


@contextmanager
def _baseline_mode() -> Generator[None, None, None]:
    """Temporarily disable retrieval and LoRA for a baseline generation."""

    _boot._STATE_LOCK.acquire()
    try:
        if _boot.BASELINE_CHAT is None:
            _boot.BASELINE_CHAT = _load_baseline_chat()

        old_values = (
            _boot.CHAT,
            _boot.INDEX,
            _boot.TEXTS,
            _boot.URLS,
            _boot.EMBEDDER,
            _boot.RERANKER,
            _boot._RETRIEVAL_DISABLED,
        )
        _boot.CHAT = _boot.BASELINE_CHAT
        _boot.INDEX = _boot.TEXTS = _boot.URLS = _boot.EMBEDDER = _boot.RERANKER = None
        _boot._RETRIEVAL_DISABLED = True

        yield
    finally:
        (
            _boot.CHAT,
            _boot.INDEX,
            _boot.TEXTS,
            _boot.URLS,
            _boot.EMBEDDER,
            _boot.RERANKER,
            _boot._RETRIEVAL_DISABLED,
        ) = old_values
        _boot._STATE_LOCK.release()


__all__: list[str] = []
