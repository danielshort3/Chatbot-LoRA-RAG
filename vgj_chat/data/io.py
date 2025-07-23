from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

import logging

import faiss  # type: ignore

from ..config import CFG

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


logger = logging.getLogger(__name__)


def load_index(path: Path) -> faiss.Index:
    """Load a FAISS index from *path* using GPU when available."""

    index = faiss.read_index(str(path))

    if CFG.cuda and getattr(faiss, "get_num_gpus", lambda: 0)() > 0:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as exc:  # pragma: no cover - GPU runtime errors
            logger.warning("Falling back to CPU index due to GPU error: %s", exc)

    return index


def load_metadata(path: Path) -> Tuple[List[str], List[str]]:
    """Return cleaned texts and URLs from a JSONL metadata file."""
    raw_meta = [json.loads(l) for l in path.read_text().splitlines()]
    texts = [clean(m["text"]) for m in raw_meta]
    urls = [m["url"] for m in raw_meta]
    return texts, urls
