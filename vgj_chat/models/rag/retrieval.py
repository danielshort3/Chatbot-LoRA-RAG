from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import boot as _boot
from .boot import logger

DOC_TOP_K = 50
WIN_TOP_K = 3
MMR_LAMBDA = 0.3


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


@dataclass
class _Window:
    doc_id: int
    para_id: int
    url: str
    date: str
    text: str


class SentenceWindowRetriever:
    """Two‑stage retriever operating on 3‑sentence windows."""

    def __init__(
        self,
        doc_top_k: int = DOC_TOP_K,
        win_top_k: int = WIN_TOP_K,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> None:
        self.doc_top_k = doc_top_k
        self.win_top_k = win_top_k
        self.mmr_lambda = mmr_lambda

    _SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")

    @classmethod
    def _windows_from_doc(cls, text: str) -> List[Tuple[int, str]]:
        windows: List[Tuple[int, str]] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p_idx, para in enumerate(paragraphs):
            sentences = [s.strip() for s in cls._SENT_SPLIT_RX.split(para) if s.strip()]
            if not sentences:
                continue
            for i in range(len(sentences)):
                win = sentences[i : i + 3]
                if not win:
                    continue
                windows.append((p_idx, " ".join(win)))
                if i + 3 >= len(sentences):
                    break
        return windows

    def retrieve_windows(self, query: str) -> List[str]:
        if _boot._RETRIEVAL_DISABLED:
            return []
        _boot._ensure_boot()
        assert _boot.EMBEDDER and _boot.INDEX and _boot.TEXTS and _boot.URLS

        logger.debug("🔍 Query: %s", query)
        q_vec = _boot.EMBEDDER.encode(query, normalize_embeddings=True).astype(
            "float32"
        )[None, :]
        _d, idx = _boot.INDEX.search(q_vec, self.doc_top_k)

        windows: List[_Window] = []
        for doc_id in idx[0]:
            text = _boot.TEXTS[doc_id]
            url = _boot.URLS[doc_id]
            for para_id, win_text in self._windows_from_doc(text):
                windows.append(_Window(doc_id, para_id, url, "unknown", win_text))

        if not windows:
            return []

        win_texts = [w.text for w in windows]
        win_vecs = _boot.EMBEDDER.encode(win_texts, normalize_embeddings=True)
        q = q_vec[0]
        sims = [float(np.dot(vec, q)) for vec in win_vecs]

        selected: List[int] = []
        used_paras: set[Tuple[int, int]] = set()
        while len(selected) < self.win_top_k and len(selected) < len(windows):
            if not selected:
                order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
                chosen = None
                for i in order:
                    key = (windows[i].doc_id, windows[i].para_id)
                    if key not in used_paras:
                        chosen = i
                        break
                if chosen is None:
                    break
            else:
                mmr_scores: List[Tuple[float, int]] = []
                for i in range(len(windows)):
                    if i in selected:
                        continue
                    key = (windows[i].doc_id, windows[i].para_id)
                    if key in used_paras:
                        continue
                    sim_to_selected = max(
                        float(np.dot(win_vecs[i], win_vecs[j])) for j in selected
                    )
                    mmr = (
                        self.mmr_lambda * sims[i]
                        - (1 - self.mmr_lambda) * sim_to_selected
                    )
                    mmr_scores.append((mmr, i))
                if not mmr_scores:
                    break
                chosen = max(mmr_scores, key=lambda x: x[0])[1]

            selected.append(chosen)
            used_paras.add((windows[chosen].doc_id, windows[chosen].para_id))

        blocks = [
            (
                f"<DOC_ID:{windows[i].doc_id}> <PARA_ID:{windows[i].para_id}> "
                f"<URL:{windows[i].url}> <DATE:{windows[i].date}>\n{windows[i].text}"
            )
            for i in selected
        ]
        return blocks


_DEFAULT_SENTENCE_WINDOW_RETRIEVER = SentenceWindowRetriever()


def retrieve_windows(query: str) -> List[str]:
    """Return top windows with metadata tags for *query*."""

    return _DEFAULT_SENTENCE_WINDOW_RETRIEVER.retrieve_windows(query)


__all__ = [
    "retrieve_unique",
    "SentenceWindowRetriever",
    "retrieve_windows",
]
