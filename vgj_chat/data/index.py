"""Utilities to chunk HTML text and build a FAISS index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss  # type: ignore
import nltk
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

CHUNK_TOKENS = 200
OVERLAP_TOKENS = 40


def chunks(
    text: str, max_tok: int = CHUNK_TOKENS, ov: int = OVERLAP_TOKENS
) -> Iterable[str]:
    """Yield sentence-overlapping chunks of roughly ``max_tok`` words."""
    sents = nltk.sent_tokenize(text)
    buf: list[str] = []
    cur = 0
    for s in sents:
        n = len(s.split())
        if cur + n > max_tok and buf:
            yield " ".join(buf)
            buf = buf[-ov:] if ov else []
            cur = sum(len(t.split()) for t in buf)
        buf.append(s)
        cur += n
    if buf:
        yield " ".join(buf)


def build_index(
    txt_dir: Path, index_path: Path, meta_path: Path, embed_model: str
) -> None:
    """Chunk ``txt_dir`` and build a FAISS index + metadata file."""
    embedder = SentenceTransformer(embed_model)
    index = None
    meta_f = meta_path.open("w")
    files = sorted(txt_dir.glob("*.txt"))
    for f in tqdm(files, desc="chunk->embed->index", unit="doc"):
        url = (f.parent / f"{f.stem}.url").read_text().strip()
        if url.startswith("https://www.visitgrandjunction.com/blog/all-posts"):
            continue
        text = f.read_text()
        for chunk in chunks(text):
            vec = embedder.encode(
                [chunk], convert_to_numpy=True, normalize_embeddings=True
            )[0]
            if index is None:
                index = faiss.IndexFlatIP(vec.shape[0])
            index.add(vec.reshape(1, -1))
            meta_f.write(json.dumps({"url": url, "text": chunk}) + "\n")
    meta_f.close()
    if index is not None:
        faiss.write_index(index, str(index_path))
