"""Utility to build the FAISS index used for retrieval."""

import argparse
from pathlib import Path

import nltk

from vgj_chat.data.index import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to index",
    )
    args = parser.parse_args()

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
        max_docs=args.limit,
    )
