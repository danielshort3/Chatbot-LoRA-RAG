"""Utility to build the FAISS index used for retrieval."""

from pathlib import Path

import nltk

from vgj_chat.data.index import build_index

if __name__ == "__main__":
    for res in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    build_index(
        Path("data/html_txt"),
        Path("faiss.index"),
        Path("meta.jsonl"),
        "sentence-transformers/all-MiniLM-L6-v2",
    )
