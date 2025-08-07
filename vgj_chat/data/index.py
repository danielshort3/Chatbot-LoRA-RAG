"""Utilities to build a FAISS index over sentence windows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import faiss  # type: ignore
import nltk
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def windowize(text: str, size: int = 3, stride: int = 1) -> Iterable[str]:
    """Yield sentence windows of ``size`` with ``stride`` overlap."""
    sents = nltk.sent_tokenize(text)
    if not sents:
        return
    if len(sents) <= size:
        yield " ".join(sents)
        return
    for i in range(0, len(sents) - size + 1, stride):
        yield " ".join(sents[i : i + size])


def build_index(
    txt_dir: Path,
    index_path: Path,
    meta_path: Path,
    embed_model: str,
    max_docs: int | None = None,
) -> None:
    """Windowize ``txt_dir`` and build a FAISS index + metadata file."""
    for res in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:  # pragma: no cover - depends on user env
            nltk.download(res, quiet=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Using %s for embeddings (CUDA available: %s)",
        device,
        torch.cuda.is_available(),
    )
    embedder = SentenceTransformer(embed_model, device=device)
    index = None
    meta_f = meta_path.open("w")
    files = sorted(txt_dir.glob("*.txt"))
    doc_id = 0
    for f in tqdm(files, desc="window->embed->index", unit="doc"):
        url = (f.parent / f"{f.stem}.url").read_text().strip()
        if url.startswith("https://www.visitgrandjunction.com/blog/all-posts"):
            continue
        text = f.read_text()
        windows = list(windowize(text))
        if not windows:
            continue
        vecs = embedder.encode(
            windows,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for window_idx, (win_text, vec) in enumerate(zip(windows, vecs)):
            if index is None:
                index = faiss.IndexFlatIP(vec.shape[0])
            index.add(vec.reshape(1, -1))
            meta_f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "window_idx": window_idx,
                        "text": win_text,
                        "url": url,
                    }
                )
                + "\n"
            )
        doc_id += 1
        if max_docs is not None and doc_id >= max_docs:
            break
    meta_f.close()
    if index is not None:
        faiss.write_index(index, str(index_path))
