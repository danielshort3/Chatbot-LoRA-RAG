from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore


FAQ_RX = re.compile(r"^[QA]:", re.I)
FOOTER_RX = re.compile(
    r"(visit\s+grand\s+junction\s+is|©|all\s+rights\s+reserved|privacy\s+policy)",
    re.I,
)


def clean(text: str) -> str:
    """Remove FAQ markers and boiler-plate footer lines."""
    cleaned_lines = []
    for line in text.splitlines():
        if FAQ_RX.match(line.strip()):
            continue
        if FOOTER_RX.search(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def load_index(path: Path) -> faiss.Index:
    """Load a FAISS index from *path*."""
    return faiss.read_index(str(path))


def load_metadata(path: Path) -> Tuple[List[str], List[str]]:
    """Return cleaned texts and URLs from a JSONL metadata file."""
    raw_meta = [json.loads(l) for l in path.read_text().splitlines()]
    texts = [clean(m["text"]) for m in raw_meta]
    urls = [m["url"] for m in raw_meta]
    return texts, urls
