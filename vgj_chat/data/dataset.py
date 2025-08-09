"""Auto-generate Q&A pairs from crawled pages using an Ollama model.

Now chunks each page into ≤256-token sections that start/end on natural
boundaries (sentences, headings, bullets) so each sample is a complete thought.
Also avoids duplicate outputs and shows a per-page sections tqdm.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
import hashlib
from pathlib import Path
from typing import List, Tuple

import requests
import trafilatura
from tqdm.auto import tqdm

from ..config import CFG
from ..utils.text import token_len, clean_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Name of the model to query from Ollama.  The container running the dataset
# build is expected to have the model pulled locally.
LLM_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# API endpoint for the running Ollama server.
API_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# Where Ollama looks for model blobs.  Useful for debugging when the model
# cannot be located inside the container.
MODELS_DIR = Path(os.getenv("OLLAMA_MODELS", "/root/.ollama"))

# Enable verbose diagnostics by setting ``OLLAMA_DEBUG=1``.
DEBUG = os.getenv("OLLAMA_DEBUG", "0") == "1"
SHOW_SECTION_TQDM = os.getenv("SHOW_SECTION_TQDM", "1") == "1"


def _dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# How long the server should keep the model loaded between requests.  This
# ensures the model stays in memory for the duration of the dataset build and
# can be unloaded explicitly afterwards.
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# How long to wait for the Ollama server to become responsive when started.
STARTUP_TIMEOUT = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "60"))

# (Legacy) Maximum number of paragraphs to consider per page when generating
# a question and answer pair. Kept for compatibility; no longer used.
PARA_MAX = 3  # unused (left for backward-compatibility)

# Target cap for answer tokens (not strictly enforced here; kept for reference).
ANSWER_TOK_CAP = 220

# Maximum number of context snippets to prepend to each question.
CTX_MAX = 3

# Chunking: hard cap requested by user.
MAX_CHUNK_TOKENS = 256
# Soft minimum to avoid tiny fragments; the chunker tries to reach this
# unless it would exceed MAX_CHUNK_TOKENS or we run out of material.
MIN_CHUNK_TOKENS = 96

# Paragraph filtering
MIN_PARA_WORDS = int(os.getenv("MIN_PARA_WORDS", "25"))

TXT_DIR = Path("data/html_txt")
RAW_HTML_DIR = Path("data/raw_html")

# store auto-generated pairs under data/dataset/
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def _ollama_generate(prompt: str) -> str:
    """Send ``prompt`` to the Ollama server and return the model response."""

    payload = {
        "model": LLM_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
    }
    r = requests.post(f"{API_URL.rstrip('/')}/api/generate", json=payload, timeout=None)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def _wait_for_server(proc: subprocess.Popen) -> bool:
    """Return ``True`` when the Ollama server is ready to accept requests."""

    version_ep = f"{API_URL.rstrip('/')}/api/version"
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(version_ep, timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _start_server() -> subprocess.Popen:
    """Launch ``ollama serve`` and wait for it to become responsive."""
    try:
        proc = subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Could not find 'ollama'. Install Ollama and ensure it is on PATH."
        ) from e

    if not _wait_for_server(proc):
        proc.terminate()
        raise RuntimeError("Ollama server failed to start")
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    """Terminate the ``ollama serve`` process."""

    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _stop_model() -> None:
    """Best-effort attempt to unload the model from the Ollama server."""

    try:
        requests.post(
            f"{API_URL.rstrip('/')}/api/stop", json={"model": LLM_NAME}, timeout=5
        )
    except Exception:
        pass


def _ensure_model() -> None:
    """Verify ``LLM_NAME`` is present in the local Ollama models directory."""

    try:
        r = requests.get(f"{API_URL.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as exc:  # pragma: no cover - network issues
        raise RuntimeError("Failed to query local Ollama models") from exc

    models = r.json().get("models", [])
    for m in models:
        if (m.get("name") or m.get("model")) == LLM_NAME:
            _dprint(f"Found model '{LLM_NAME}' via /api/tags")
            return
    available = [m.get("name") or m.get("model") for m in models]
    _dprint(f"Available models reported by Ollama: {available}")
    _dprint(f"Expecting models directory at: {MODELS_DIR}")
    _dprint(f"Models directory exists: {MODELS_DIR.exists()}")
    raise RuntimeError(
        (
            f"Model '{LLM_NAME}' not found. Available models: {available or 'none'}\n"
            f"Ollama models dir: {MODELS_DIR} (exists: {MODELS_DIR.exists()})\n"
            "Mount your host models directory to /root/.ollama (for example:\n"
            "  -v $HOME/.ollama:/root/.ollama or\n"
            "  -v /mnt/c/Users/<user>/.ollama/models:/root/.ollama) and pull it with\n"
            f"'ollama pull {LLM_NAME}'."
        )
    )


# ---------------------------------------------------------------------------
# Text generation utilities
# ---------------------------------------------------------------------------


def _gen_question(passage: str) -> str:
    sys = (
        "You are a helpful travel assistant. Read the PASSAGE and invent one "
        "concise, natural-sounding traveler question that could be answered "
        "by the same passage. Return ONLY the question text."
    )
    prompt = f"{sys}\nPASSAGE:\n'''{passage}'''"
    q = _ollama_generate(prompt)
    return q if q.endswith("?") else q + "?"


def _gen_context(question: str, answer_part: str) -> str:
    """Generate a short context passage supporting ``answer_part``."""

    sys = (
        "You are a helpful travel assistant. Invent a short context passage of "
        "2-3 sentences that could help answer the QUESTION, focusing on the "
        "ANSWER_SNIPPET. Do not answer the question directly."
    )
    prompt = f"{sys}\nQUESTION: {question}\nANSWER_SNIPPET: {answer_part}"
    for _ in range(3):
        ctx = _ollama_generate(prompt)
        sentences = re.findall(r"[^.!?]+[.!?]", ctx)
        if len(sentences) >= 2:
            keep = min(len(sentences), random.choice([2, 3]))
            return " ".join(s.strip() for s in sentences[:keep])
    return ctx


def _choose_num_ctx(max_ctx: int) -> int:
    """Sample how many context snippets to include for a question."""
    num = round(random.gauss(2.0, 0.8))
    num = max(0, min(3, num))
    return min(num, max_ctx)


BOILER_PAT = re.compile(
    r"(click here|minute read|photo credit|browser is not supported)", re.I
)


# ---------------------------------------------------------------------------
# Chunking helpers (≤256 tokens, sentence/heading/bullet aligned)
# ---------------------------------------------------------------------------


def _is_heading_line(line: str) -> bool:
    """Heuristic to detect headings or section titles."""
    s = line.strip()
    if not s:
        return False
    # Short, no end punctuation, often Title Case or ALL CAPS or ends with colon.
    if len(s.split()) <= 12 and not re.search(r"[.!?]$", s):
        if s.endswith(":") or s.istitle() or s.isupper():
            return True
    return False


def _is_bullet_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*•\u2022]|\d+[.)])\s+", line))


def _sentences_from_paragraph(p: str) -> list[str]:
    """Split a paragraph into sentence-like units, preserving bullets/headings."""
    p = p.strip()
    if not p:
        return []
    if _is_bullet_line(p) or _is_heading_line(p):
        return [p]

    # Capture sentences with terminal punctuation, keeping trailing quotes/brackets.
    parts = re.findall(r"[^.!?]+[.!?]+(?:['\"”’)\]]+)?", p)
    if not parts:
        return [p]

    # If there's any trailing fragment without terminal punctuation, keep it.
    consumed = "".join(parts)
    tail = p[len(consumed) :].strip()
    if tail:
        parts.append(tail)

    return [s.strip() for s in parts if s.strip()]


def _split_overlong_unit(unit: str, max_tokens: int) -> list[str]:
    """Split a single overlong unit (rare) by clause/word boundaries."""
    if token_len(unit) <= max_tokens:
        return [unit]

    # Prefer clause boundaries first.
    clauses = re.split(r"(?<=[;:—–\-])\s+|,\s+", unit)
    parts: list[str] = []
    buf: list[str] = []
    for c in clauses:
        cand = (" ".join(buf + [c])).strip()
        if token_len(cand) <= max_tokens:
            buf.append(c)
        else:
            if buf:
                parts.append(" ".join(buf).strip())
                buf = [c]
            else:
                # Fall back to word-splitting if even a clause is too large.
                words = c.split()
                step = max(1, max_tokens - 8)
                for i in range(0, len(words), step):
                    seg = " ".join(words[i : i + step]).strip()
                    if seg:
                        parts.append(seg)
                buf = []
    if buf:
        parts.append(" ".join(buf).strip())

    # Last safety pass
    safe: list[str] = []
    for p in parts:
        if token_len(p) <= max_tokens:
            safe.append(p)
        else:
            words = p.split()
            step = max(1, max_tokens - 8)
            for i in range(0, len(words), step):
                seg = " ".join(words[i : i + step]).strip()
                if seg:
                    safe.append(seg)
    return safe


def _units_from_paragraphs(paras: list[str], max_tokens: int) -> list[Tuple[str, str]]:
    """Turn paragraphs into a flat sequence of units (text, kind)."""
    units: list[Tuple[str, str]] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if _is_bullet_line(p):
            candidates = [p]
        elif _is_heading_line(p):
            candidates = [p]
        else:
            candidates = _sentences_from_paragraph(p)

        for u in candidates:
            kind = (
                "bullet" if _is_bullet_line(u) else "heading" if _is_heading_line(u) else "sent"
            )
            for seg in _split_overlong_unit(u, max_tokens):
                units.append((seg, kind))
    return units


def _ends_awkwardly(text: str) -> bool:
    """Avoid stopping on dangling punctuation that implies continuation."""
    return bool(re.search(r"[:;,\-–—]\s*$", text))


def _chunk_paragraphs(paras: list[str], max_tokens: int) -> list[list[str]]:
    """Chunk paragraphs into lists of units where each chunk ≤ ``max_tokens`` tokens.

    Heuristics:
      - Never break inside a sentence/bullet/heading.
      - Try not to end on dangling punctuation (colon/dash/comma).
      - Try to reach a soft minimum (MIN_CHUNK_TOKENS) unless near the end.
      - Include a heading with the following content when possible.
    """
    units = _units_from_paragraphs(paras, max_tokens)
    chunks: list[list[str]] = []
    i = 0
    n = len(units)

    while i < n:
        chunk: list[str] = []
        cur_tokens = 0
        j = i

        # If we're at a heading, include it first.
        if j < n and units[j][1] == "heading":
            u_text = units[j][0]
            t = token_len(u_text)
            chunk.append(u_text)
            cur_tokens += t
            j += 1

        # Greedily add subsequent units up to the cap.
        while j < n:
            u_text, _ = units[j]
            t = token_len(u_text)
            if cur_tokens == 0:
                chunk.append(u_text)
                cur_tokens += t
                j += 1
                continue
            if cur_tokens + t <= max_tokens:
                chunk.append(u_text)
                cur_tokens += t
                j += 1
            else:
                break

        # Avoid awkward endings if possible.
        if chunk and _ends_awkwardly(chunk[-1]) and j < n:
            nxt_t = token_len(units[j][0])
            if cur_tokens + nxt_t <= max_tokens:
                chunk.append(units[j][0])
                cur_tokens += nxt_t
                j += 1
            else:
                if len(chunk) > 1 and not _is_heading_line(chunk[-1]):
                    cur_tokens -= token_len(chunk[-1])
                    chunk.pop()

        # Try to hit a soft minimum for chunk size.
        while cur_tokens < MIN_CHUNK_TOKENS and j < n:
            nxt_t = token_len(units[j][0])
            if cur_tokens + nxt_t <= max_tokens:
                chunk.append(units[j][0])
                cur_tokens += nxt_t
                j += 1
            else:
                break

        if chunk:
            chunk_text = "\n\n".join(chunk).strip()
            if chunk_text and not BOILER_PAT.search(chunk_text):
                chunks.append(chunk)

        i = j if j > i else i + 1

    # Merge a very short tail with the previous chunk if it fits cleanly.
    if len(chunks) >= 2:
        last = "\n\n".join(chunks[-1])
        if token_len(last) < MIN_CHUNK_TOKENS:
            prev = "\n\n".join(chunks[-2])
            if token_len(prev + "\n\n" + last) <= max_tokens:
                chunks[-2].extend(chunks[-1])
                chunks.pop()

    # Final hard-cap safety
    trimmed: list[list[str]] = []
    for ch in chunks:
        text = "\n\n".join(ch)
        if token_len(text) <= max_tokens:
            trimmed.append(ch)
            continue
        # Trim units off the end until it fits
        cur: list[str] = []
        total = 0
        for u in ch:
            t = token_len(u)
            if total + t <= max_tokens or not cur:
                cur.append(u)
                total += t
            else:
                break
        trimmed.append(cur)
    return trimmed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint(s: str) -> str:
    """Stable, normalized fingerprint for deduplication."""
    norm = re.sub(r"\s+", " ", clean_text(s)).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _collect_passages_by_page() -> tuple[list[tuple[str, list[list[str]]]], int, int]:
    """Return ([(page_id, [chunks])...], skipped_pages, long_outputs)."""
    pages_chunks: list[tuple[str, list[list[str]]]] = []
    skipped_pages = 0
    long_outputs = 0

    # Honor CFG.max_new_tokens as a hard ceiling if it's smaller than 256.
    try:
        max_allowed = int(getattr(CFG, "max_new_tokens", MAX_CHUNK_TOKENS))
    except Exception:
        max_allowed = MAX_CHUNK_TOKENS
    effective_max = min(MAX_CHUNK_TOKENS, max_allowed)

    for txt_f in sorted(TXT_DIR.glob("*.txt")):
        page_id = txt_f.stem
        html_path = RAW_HTML_DIR / f"{page_id}.html"
        if not html_path.exists():
            skipped_pages += 1
            continue

        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            skipped_pages += 1
            continue

        text = trafilatura.extract(html) or ""
        raw_paras = [p.strip() for p in text.splitlines()]
        paras = [p for p in raw_paras if len(p.split()) >= MIN_PARA_WORDS]

        if not paras:
            skipped_pages += 1
            continue

        page_chunks = _chunk_paragraphs(paras, effective_max)

        # Extra safety against generator cap (unlikely at 256).
        valid_chunks: list[list[str]] = []
        for ch in page_chunks:
            ch_text = "\n\n".join(ch)
            if token_len(ch_text) > max_allowed:
                long_outputs += 1
                continue
            valid_chunks.append(ch)

        if not valid_chunks:
            skipped_pages += 1
            continue

        pages_chunks.append((page_id, valid_chunks))

    return pages_chunks, skipped_pages, long_outputs


def count_expected_pairs() -> int:
    """Return the number of QA pairs expected after boilerplate filtering.

    Note: this equals the number of *chunks*, not pages.
    """
    pages_chunks, _, _ = _collect_passages_by_page()
    return sum(len(chunks) for _, chunks in pages_chunks)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_auto_dataset() -> None:
    pages_chunks, skipped, long_outputs = _collect_passages_by_page()

    total_chunks = sum(len(chunks) for _, chunks in pages_chunks)
    if total_chunks == 0:
        print(
            f"No passages within token limit; skipped {skipped} pages "
            f"(boilerplate/empty) and {long_outputs} passages over "
            f"{getattr(CFG, 'max_new_tokens', 'N/A')} tokens."
        )
        return

    # If dataset exists with enough samples, skip rebuild.
    existing: list[dict[str, str]] = []
    if AUTO_QA_JL.exists():
        try:
            for line in AUTO_QA_JL.read_text(encoding="utf-8").splitlines():
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        except OSError:
            pass
        if existing and len(existing) >= total_chunks:
            print(
                f"{AUTO_QA_JL} exists with {len(existing)} samples; skipping dataset build"
            )
            return

    AUTO_QA_JL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting Ollama server for {LLM_NAME} …")
    server = _start_server()
    _ensure_model()

    # Dedup across the whole run.
    seen_outputs: set[str] = set()
    seen_pairs: set[str] = set()

    generated = 0
    try:
        with AUTO_QA_JL.open("w", encoding="utf-8") as f:
            pages_bar = tqdm(pages_chunks, desc="pages", unit="page")
            for page_id, chunks in pages_bar:
                if not chunks:
                    continue

                if SHOW_SECTION_TQDM:
                    sec_bar = tqdm(
                        total=len(chunks),
                        desc=f"{page_id} sections",
                        unit="sec",
                        leave=False,
                    )
                else:
                    sec_bar = None

                for chunk_blocks in chunks:
                    # Build passage (≤256 tokens by construction)
                    passage = clean_text("\n\n".join(chunk_blocks))
                    # Double-check hard cap at write time
                    if token_len(passage) > MAX_CHUNK_TOKENS:
                        # Trim last unit(s) until it fits
                        cur: list[str] = []
                        total = 0
                        for blk in chunk_blocks:
                            t = token_len(blk)
                            if total + t <= MAX_CHUNK_TOKENS or not cur:
                                cur.append(blk)
                                total += t
                            else:
                                break
                        passage = clean_text("\n\n".join(cur))

                    # Generate question & optional contexts (1 sample per chunk)
                    question = clean_text(_gen_question(passage))
                    available_parts = [clean_text(p) for p in chunk_blocks]

                    ctx_blocks: list[str] = []
                    if available_parts:
                        max_ctx = min(len(available_parts), CTX_MAX)
                        num_ctx = _choose_num_ctx(max_ctx)
                        if num_ctx:
                            for part in random.sample(available_parts, k=num_ctx):
                                ctx_blocks.append(
                                    f"<CONTEXT>\n{clean_text(_gen_context(question, part))}\n</CONTEXT>"
                                )
                            random.shuffle(ctx_blocks)

                    ctx_str = "\n\n".join(ctx_blocks)
                    prompt = clean_text(f"{ctx_str}\n\n{question}" if ctx_blocks else question)
                    answer_clean = clean_text(passage)

                    # Deduplicate (prefer strong output-dedup, and pair-dedup).
                    out_fp = _fingerprint(answer_clean)
                    if out_fp in seen_outputs:
                        if sec_bar:
                            sec_bar.update(1)
                        continue  # skip duplicate output

                    pair_fp = _fingerprint(prompt + "\n\n==>\n\n" + answer_clean)
                    if pair_fp in seen_pairs:
                        if sec_bar:
                            sec_bar.update(1)
                        continue

                    f.write(
                        json.dumps({"input": prompt, "output": answer_clean}, ensure_ascii=False)
                        + "\n"
                    )
                    seen_outputs.add(out_fp)
                    seen_pairs.add(pair_fp)
                    generated += 1

                    if sec_bar:
                        sec_bar.update(1)

                if sec_bar:
                    sec_bar.close()

        print(
            f"Generated {generated:,} unique samples → {AUTO_QA_JL}; "
            f"skipped {skipped} pages (boilerplate/empty); "
            f"skipped {long_outputs} chunks over {getattr(CFG, 'max_new_tokens', 'N/A')} tokens.",
        )
    finally:
        _stop_model()
        _stop_server(server)


__all__ = [
    "build_auto_dataset",
    "_gen_question",
    "_gen_context",
    "_choose_num_ctx",
    "count_expected_pairs",
]