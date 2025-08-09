"""Text utility helpers."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RX = re.compile(r"\s+")
_EXTRA_RX = re.compile(r"^(?:question|answer|reference|sources?):", re.IGNORECASE)

_WS_RX = re.compile(r"[ \t\r\f\v]+")
_CLEAN_TRANSLATION = {
    ord("\u2018"): "'",  # left single quotation mark
    ord("\u2019"): "'",  # right single quotation mark
    ord("\u201c"): '"',  # left double quotation mark
    ord("\u201d"): '"',  # right double quotation mark
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u00a0"): " ",  # non-breaking space
    ord("\u202f"): " ",  # narrow no-break space
    ord("\u200b"): "",  # zero width space
}


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


def clean_text(text: str) -> str:
    """Normalize unicode and strip problem characters from *text*.

    This helper replaces common typographic quotes, dashes, and no-break
    spaces with their ASCII equivalents. It also collapses runs of
    whitespace (except for newlines) into a single space.
    """

    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CLEAN_TRANSLATION)
    text = _WS_RX.sub(" ", text)
    return text.strip()


__all__ = ["token_len", "strip_metadata", "clean_text"]
