"""Auto-generate Q&A pairs from crawled pages using an Ollama model."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import requests
import trafilatura
from tqdm.auto import tqdm
from ..utils.text import token_len

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


def _dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# How long the server should keep the model loaded between requests.  This
# ensures the model stays in memory for the duration of the dataset build and
# can be unloaded explicitly afterwards.
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# How long to wait for the Ollama server to become responsive when started.
STARTUP_TIMEOUT = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "60"))

# Maximum number of paragraphs to consider per page when generating a question
# and answer pair.  These constants mirror the previous implementation to keep
# the dataset format stable.
PARA_MAX = 3
ANSWER_TOK_CAP = 220
CTX_MAX = 5


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
    """Verify ``LLM_NAME`` is present in the local Ollama models directory.

    The dataset builder expects the model to be pre-pulled on the host and the
    directory mounted into the container (typically ``-v ~/.ollama:/root/.ollama``).
    If the model cannot be found a clear ``RuntimeError`` is raised so callers
    know to pull it manually instead of silently downloading gigabytes again.
    """

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

    num = round(random.gauss(2.5, 1))
    num = max(0, min(5, num))
    return min(num, max_ctx)


BOILER_PAT = re.compile(
    r"(click here|minute read|photo credit|browser is not supported)", re.I
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_passages() -> tuple[list[list[str]], int]:
    """Return paragraphs for each non-boiler page and a skip count."""

    passages: list[list[str]] = []
    skipped = 0
    for txt_f in sorted(TXT_DIR.glob("*.txt")):
        html = (RAW_HTML_DIR / f"{txt_f.stem}.html").read_text()
        text = trafilatura.extract(html) or ""
        paras = [p.strip() for p in text.splitlines() if len(p.split()) > 25][:PARA_MAX]
        if not paras:
            skipped += 1
            continue
        passage = "\n\n".join(paras)
        if BOILER_PAT.search(passage):
            skipped += 1
            continue
        passages.append(paras)
    return passages, skipped


def count_expected_pairs() -> int:
    """Return the number of QA pairs expected after boilerplate filtering."""

    passages, _ = _collect_passages()
    return len(passages)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_auto_dataset() -> None:
    passages, skipped = _collect_passages()

    start_idx = 0
    valid_examples: list[dict[str, str]] = []

    if AUTO_QA_JL.exists():
        try:
            lines = AUTO_QA_JL.read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                valid_examples.append(json.loads(line))
            except json.JSONDecodeError:
                break
        qa_count = len(valid_examples)
        if qa_count == len(passages) and qa_count > 0:
            print(f"{AUTO_QA_JL} exists with {qa_count} pairs; skipping dataset build")
            return
        if qa_count:
            # drop the last entry to ensure it is regenerated
            valid_examples = valid_examples[:-1]
            start_idx = len(valid_examples)
            with AUTO_QA_JL.open("w") as f:
                for ex in valid_examples:
                    f.write(json.dumps(ex) + "\n")
        print(f"{AUTO_QA_JL} incomplete; resuming dataset build")

    AUTO_QA_JL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting Ollama server for {LLM_NAME} …")
    server = _start_server()
    _ensure_model()
    generated = 0
    long_outputs = 0
    try:
        with AUTO_QA_JL.open("a") as f:
            for paras in tqdm(passages[start_idx:], desc="auto-QA", unit="page"):
                passage = "\n\n".join(paras)
                words, answer_words, used_paras = 0, [], []
                for p in paras:
                    if words + len(p.split()) > ANSWER_TOK_CAP:
                        break
                    answer_words.extend(p.split())
                    words += len(p.split())
                    used_paras.append(p)
                answer = " ".join(answer_words) or paras[0]
                if token_len(answer) > 512:
                    long_outputs += 1
                    continue
                question = _gen_question(passage)
                available_parts = used_paras
                ctx_blocks: list[str] = []
                if available_parts:
                    max_ctx = min(len(available_parts), CTX_MAX)
                    num_ctx = _choose_num_ctx(max_ctx)
                    if num_ctx:
                        ctx_parts = random.sample(available_parts, k=num_ctx)
                        for part in ctx_parts:
                            ctx_blocks.append(
                                f"<CONTEXT>\n{_gen_context(question, part)}\n</CONTEXT>",
                            )
                        random.shuffle(ctx_blocks)
                ctx_str = "\n\n".join(ctx_blocks)
                prompt = f"{ctx_str}\n\n{question}" if ctx_blocks else question
                f.write(json.dumps({"input": prompt, "output": answer}) + "\n")
                generated += 1
        print(
            f"Generated {generated:,} new pairs → {AUTO_QA_JL}; skipped {skipped} passages; "
            f"{long_outputs} outputs over 512 tokens",
        )
    finally:
        # Ensure the model is unloaded and server stopped even if generation fails.
        _stop_model()
        _stop_server(server)


__all__ = [
    "build_auto_dataset",
    "_gen_question",
    "_gen_context",
    "_choose_num_ctx",
    "count_expected_pairs",
]
