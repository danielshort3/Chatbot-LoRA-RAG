"""Text utility helpers."""

from __future__ import annotations

import re


_TOKEN_RX = re.compile(r"\s+")


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
