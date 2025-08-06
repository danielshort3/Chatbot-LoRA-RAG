from __future__ import annotations

from typing import List, Tuple

from . import boot as _boot
from .boot import logger


def retrieve_unique(query: str) -> List[Tuple[float, str, str]]:
    """Return the top-K unique passages for *query* sorted by score."""

    if _boot._RETRIEVAL_DISABLED:
        return []
    _boot._ensure_boot()
    assert (
        _boot.EMBEDDER and _boot.INDEX and _boot.TEXTS and _boot.URLS and _boot.RERANKER
    )

    logger.debug("🔍 Query: %s", query)

    q_vec = _boot.EMBEDDER.encode(query, normalize_embeddings=True).astype("float32")[
        None, :
    ]
    _d, idx = _boot.INDEX.search(q_vec, 100)

    candidates = [(_boot.TEXTS[i], _boot.URLS[i]) for i in idx[0]]
    raw_scores = _boot.RERANKER.predict([(query, t) for t, _ in candidates])

    best: dict[str, Tuple[float, str]] = {}
    for score, (text, url) in zip(raw_scores, candidates):
        if score < _boot.CFG.score_min:
            continue
        best[url] = max(best.get(url, (0, "")), (score, text))

    uniques = sorted(
        ((s, t, u) for u, (s, t) in best.items()),
        key=lambda x: x[0],
        reverse=True,
    )[: _boot.CFG.top_k]

    logger.debug("Retrieved %d unique passages.", len(uniques))
    return uniques


__all__ = ["retrieve_unique"]
