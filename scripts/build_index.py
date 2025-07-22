from pathlib import Path

from vgj_chat.data.index import CHUNK_TOKENS, OVERLAP_TOKENS, build_index

if __name__ == "__main__":
    build_index(
        Path("data/html_txt"),
        Path("faiss.index"),
        Path("meta.jsonl"),
        "sentence-transformers/all-MiniLM-L6-v2",
    )
