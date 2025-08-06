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
    index_path = Path("data/faiss.index")
    meta_path = Path("data/meta.jsonl")
    if index_path.exists() and meta_path.exists():
        print("FAISS index and metadata already exist; skipping build")
    else:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        build_index(
            Path("data/html_txt"),
            index_path,
            meta_path,
            "sentence-transformers/all-MiniLM-L6-v2",
            max_docs=args.limit,
        )
