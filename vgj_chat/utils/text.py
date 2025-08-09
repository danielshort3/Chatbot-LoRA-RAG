"""Text utility helpers."""

from __future__ import annotations

import re

_TOKEN_RX = re.compile(r"\s+")
_EXTRA_RX = re.compile(r"^(?:question|answer|reference|sources?):", re.IGNORECASE)


def token_len(text: str) -> int:
    """Return a naive whitespace token count for *text*.

    This helper is intentionally lightweight to avoid heavy tokenizer
    dependencies during tests. It approximates the number of tokens by
    splitting on whitespace, which is sufficient for enforcing rough
    limits on prompt assembly.
    """

    if not text:
        return 0
    return len(_TOKEN_RX.split(text.strip()))


def strip_metadata(text: str) -> str:
    """Remove extraneous sections such as references or restated Q&A."""

    if not text:
        return ""
    lines = text.strip().splitlines()
    kept: list[str] = []
    for ln in lines:
        if _EXTRA_RX.match(ln.strip()):
            break
        kept.append(ln)
    return "\n".join(kept).strip()


__all__ = ["token_len", "strip_metadata"]
