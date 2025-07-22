from __future__ import annotations

from typing import List, Tuple

import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer


def too_similar(
    answer: str,
    passages: List[Tuple[float, str, str]],
    embedder: SentenceTransformer,
    threshold: float,
) -> bool:
    """Return ``True`` if ``answer`` overlaps too closely with any passage."""
    ans_vec = embedder.encode(answer, normalize_embeddings=True).astype("float32")
    for score, text, _ in passages:
        src_vec = embedder.encode(text[:512], normalize_embeddings=True).astype(
            "float32"
        )
        cos_sim = float(np.dot(ans_vec, src_vec))
        if cos_sim >= threshold:
            return True
    return False
