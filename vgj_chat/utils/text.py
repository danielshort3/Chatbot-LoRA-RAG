"""Text utility helpers."""

from __future__ import annotations

import re
import unicodedata

# Split on runs of whitespace to approximate token count
_TOKEN_RX = re.compile(r"\s+")

# Lines that indicate extra sections we want to drop at the end
_EXTRA_RX = re.compile(r"^(?:question|answer|reference|sources?):", re.IGNORECASE)

# Collapse any whitespace except newlines into a single space
_WS_RX = re.compile(r"[^\S\n]+")

# Map common typographic Unicode characters to simple ASCII (or remove)
_CLEAN_TRANSLATION = {
    # Quotes / apostrophes
    ord("\u2018"): "'",  # left single quotation mark
    ord("\u2019"): "'",  # right single quotation mark
    ord("\u201A"): "'",  # single low-9 quotation mark
    ord("\u201B"): "'",  # single high-reversed-9 quotation mark
    ord("\u02BC"): "'",  # modifier letter apostrophe
    ord("\u2032"): "'",  # prime (often feet)
    ord("\u201C"): '"',  # left double quotation mark
    ord("\u201D"): '"',  # right double quotation mark
    ord("\u201E"): '"',  # double low-9 quotation mark
    ord("\u201F"): '"',  # double high-reversed-9 quotation mark
    ord("\u2033"): '"',  # double prime (often inches)
    ord("\u00AB"): '"',  # «
    ord("\u00BB"): '"',  # »
    ord("\u2039"): "'",  # ‹
    ord("\u203A"): "'",  # ›

    # Dashes / hyphens / minus
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign (math minus -> hyphen-minus)
    ord("\u2043"): "-",  # hyphen bullet

    # Soft hyphen (discretionary) -> drop
    ord("\u00AD"): "",

    # Ellipsis
    ord("\u2026"): "...",

    # Fraction slash
    ord("\u2044"): "/",

    # Spaces -> regular space
    ord("\u00A0"): " ",  # no-break space
    ord("\u202F"): " ",  # narrow no-break space
    ord("\u2000"): " ",  # en quad
    ord("\u2001"): " ",  # em quad
    ord("\u2002"): " ",  # en space
    ord("\u2003"): " ",  # em space
    ord("\u2004"): " ",  # three-per-em space
    ord("\u2005"): " ",  # four-per-em space
    ord("\u2006"): " ",  # six-per-em space
    ord("\u2007"): " ",  # figure space
    ord("\u2008"): " ",  # punctuation space
    ord("\u2009"): " ",  # thin space
    ord("\u200A"): " ",  # hair space
    ord("\u205F"): " ",  # medium mathematical space
    ord("\u3000"): " ",  # ideographic space

    # Zero-width / joiners / BOM -> drop
    ord("\u200B"): "",   # zero width space
    ord("\u200C"): "",   # zero width non-joiner
    ord("\u200D"): "",   # zero width joiner
    ord("\u2060"): "",   # word joiner
    ord("\uFEFF"): "",   # zero width no-break space (BOM)
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

    Replaces common typographic quotes, dashes, and no-break spaces with
    ASCII equivalents; removes zero-width characters; collapses runs of
    whitespace (except newlines) into a single space.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CLEAN_TRANSLATION)
    text = _WS_RX.sub(" ", text)
    return text.strip()


__all__ = ["token_len", "strip_metadata", "clean_text"]