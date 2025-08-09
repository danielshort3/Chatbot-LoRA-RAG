import importlib
import json
import re
import sys
import types
from types import ModuleType

import pytest


def _load_dataset(monkeypatch, text: str, tags: dict | None = None):
    """Reload dataset with requests stubbed to return ``text`` and ``tags``."""

    class DummyResp:
        ok = True

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    if tags is None:
        tags = {"models": [{"name": "gpt-oss:20b"}]}

    requests_stub = types.SimpleNamespace(
        post=lambda url, json, timeout=None: DummyResp({"response": text}),
        get=lambda url, timeout=None: DummyResp(tags),
    )
    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    monkeypatch.setitem(sys.modules, "trafilatura", ModuleType("trafilatura"))

    dataset = importlib.reload(importlib.import_module("vgj_chat.data.dataset"))
    return dataset


def test_build_auto_dataset_starts_server(monkeypatch, tmp_path):
    dataset = _load_dataset(monkeypatch, "text")

    calls = {"popen": 0, "stop": 0, "terminate": 0, "ensure": 0}

    class DummyProc:
        def poll(self):
            return None

        def terminate(self):
            calls["terminate"] += 1

        def wait(self, timeout=None):
            return None

    def popen(cmd, stdout=None, stderr=None):
        calls["popen"] += 1
        return DummyProc()

    monkeypatch.setattr(dataset.subprocess, "Popen", popen)
    monkeypatch.setattr(dataset, "_wait_for_server", lambda proc: True)
    monkeypatch.setattr(
        dataset, "_stop_model", lambda: calls.__setitem__("stop", calls["stop"] + 1)
    )
    monkeypatch.setattr(
        dataset,
        "_ensure_model",
        lambda: calls.__setitem__("ensure", calls["ensure"] + 1),
    )

    tra = ModuleType("trafilatura")
    tra.extract = lambda html: "word " * 30
    monkeypatch.setattr(dataset, "trafilatura", tra)

    monkeypatch.setattr(dataset, "TXT_DIR", tmp_path)
    monkeypatch.setattr(dataset, "RAW_HTML_DIR", tmp_path)
    (tmp_path / "page.txt").write_text("dummy")
    (tmp_path / "page.html").write_text("<p>dummy</p>")
    monkeypatch.setattr(dataset, "AUTO_QA_JL", tmp_path / "out.jsonl")

    dataset.build_auto_dataset()

    assert calls["popen"] == 1
    assert calls["terminate"] == 1
    assert calls["stop"] == 1
    assert calls["ensure"] == 1


def test_build_auto_dataset_rebuilds_when_incomplete(monkeypatch, tmp_path):
    dataset = _load_dataset(monkeypatch, "text")

    calls = {"popen": 0}

    class DummyProc:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    def popen(cmd, stdout=None, stderr=None):
        calls["popen"] += 1
        return DummyProc()

    monkeypatch.setattr(dataset.subprocess, "Popen", popen)
    monkeypatch.setattr(dataset, "_wait_for_server", lambda proc: True)
    monkeypatch.setattr(dataset, "_stop_model", lambda: None)
    monkeypatch.setattr(dataset, "_ensure_model", lambda: None)

    # provide trafilatura.extract to return enough text
    tra = ModuleType("trafilatura")
    tra.extract = lambda html: "word " * 30
    monkeypatch.setattr(dataset, "trafilatura", tra)

    # create input files and an empty existing dataset file
    monkeypatch.setattr(dataset, "TXT_DIR", tmp_path)
    monkeypatch.setattr(dataset, "RAW_HTML_DIR", tmp_path)
    txt_file = tmp_path / "page.txt"
    txt_file.write_text("dummy")
    (tmp_path / "page.html").write_text("<p>dummy</p>")
    auto_jl = tmp_path / "out.jsonl"
    auto_jl.write_text("")
    monkeypatch.setattr(dataset, "AUTO_QA_JL", auto_jl)

    dataset.build_auto_dataset()

    assert calls["popen"] == 1
    with auto_jl.open() as f:
        assert sum(1 for _ in f) == 1


def test_build_auto_dataset_skips_when_complete(monkeypatch, tmp_path):
    dataset = _load_dataset(monkeypatch, "text")

    class DummyProc:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    monkeypatch.setattr(
        dataset.subprocess,
        "Popen",
        lambda cmd, stdout=None, stderr=None: DummyProc(),
    )
    monkeypatch.setattr(dataset, "_wait_for_server", lambda proc: True)
    monkeypatch.setattr(dataset, "_stop_model", lambda: None)
    monkeypatch.setattr(dataset, "_ensure_model", lambda: None)

    tra = ModuleType("trafilatura")
    tra.extract = lambda html: "word " * 30
    monkeypatch.setattr(dataset, "trafilatura", tra)

    monkeypatch.setattr(dataset, "_gen_question", lambda passage: "Q")
    monkeypatch.setattr(dataset, "_choose_num_ctx", lambda max_ctx: 0)

    monkeypatch.setattr(dataset, "TXT_DIR", tmp_path)
    monkeypatch.setattr(dataset, "RAW_HTML_DIR", tmp_path)
    auto_jl = tmp_path / "out.jsonl"
    monkeypatch.setattr(dataset, "AUTO_QA_JL", auto_jl)

    for i in range(3):
        (tmp_path / f"page{i}.txt").write_text("dummy")
        (tmp_path / f"page{i}.html").write_text("<p>dummy</p>")

    existing = [{"input": f"existing{i}", "output": f"out{i}"} for i in range(3)]
    auto_jl.write_text("\n".join(json.dumps(e) for e in existing) + "\n")

    dataset.build_auto_dataset()

    with auto_jl.open() as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == len(existing)
    assert lines[0]["input"] == "existing0"


def test_build_auto_dataset_skips_long_passages(monkeypatch, tmp_path):
    dataset = _load_dataset(monkeypatch, "text")

    calls = {"popen": 0}

    class DummyProc:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    def popen(cmd, stdout=None, stderr=None):
        calls["popen"] += 1
        return DummyProc()

    monkeypatch.setattr(dataset.subprocess, "Popen", popen)
    monkeypatch.setattr(dataset, "_wait_for_server", lambda proc: True)
    monkeypatch.setattr(dataset, "_stop_model", lambda: None)
    monkeypatch.setattr(dataset, "_ensure_model", lambda: None)

    monkeypatch.setattr(dataset, "CFG", types.SimpleNamespace(max_new_tokens=100))
    tra = ModuleType("trafilatura")
    tra.extract = lambda html: "word " * 300
    monkeypatch.setattr(dataset, "trafilatura", tra)

    monkeypatch.setattr(dataset, "TXT_DIR", tmp_path)
    monkeypatch.setattr(dataset, "RAW_HTML_DIR", tmp_path)
    (tmp_path / "page.txt").write_text("dummy")
    (tmp_path / "page.html").write_text("<p>dummy</p>")
    auto_jl = tmp_path / "out.jsonl"
    monkeypatch.setattr(dataset, "AUTO_QA_JL", auto_jl)

    dataset.build_auto_dataset()
    assert calls["popen"] == 0
    assert not auto_jl.exists()

    dataset.build_auto_dataset()
    assert calls["popen"] == 0


def test_count_expected_pairs_filters_boiler(monkeypatch, tmp_path):
    dataset = _load_dataset(monkeypatch, "text")

    tra = ModuleType("trafilatura")
    tra.extract = lambda html: (
        "click here " + "word " * 30 if "boiler" in html else "word " * 30
    )
    monkeypatch.setattr(dataset, "trafilatura", tra)

    monkeypatch.setattr(dataset, "TXT_DIR", tmp_path)
    monkeypatch.setattr(dataset, "RAW_HTML_DIR", tmp_path)

    (tmp_path / "boiler.txt").write_text("dummy")
    (tmp_path / "boiler.html").write_text("boiler")
    (tmp_path / "good.txt").write_text("dummy")
    (tmp_path / "good.html").write_text("good")

    assert dataset.count_expected_pairs() == 1


def test_start_server_missing_binary(monkeypatch):
    dataset = _load_dataset(monkeypatch, "text")

    def popen(cmd, stdout=None, stderr=None):
        raise FileNotFoundError

    monkeypatch.setattr(dataset.subprocess, "Popen", popen)

    with pytest.raises(FileNotFoundError):
        dataset._start_server()


def test_gen_context_returns_complete_sentences(monkeypatch):
    incomplete_text = (
        "The first sentence is complete. "
        "This is another complete sentence. "
        "This final sentence is cut off"
    )
    dataset = _load_dataset(monkeypatch, incomplete_text)

    ctx = dataset._gen_context("question", "answer")

    assert ctx.endswith((".", "!", "?")), "Context should end with punctuation"
    sentences = re.findall(r"[^.!?]+[.!?]", ctx)
    assert 2 <= len(sentences) <= 3
    reconstructed = " ".join(s.strip() for s in sentences)
    assert ctx == reconstructed
    assert "cut off" not in ctx


def test_gen_context_single_sentence(monkeypatch):
    single_sentence = "Only one complete sentence here."
    dataset = _load_dataset(monkeypatch, single_sentence)

    ctx = dataset._gen_context("question", "answer")

    assert ctx.endswith((".", "!", "?"))
    assert ctx == single_sentence
    sentences = re.findall(r"[^.!?]+[.!?]", ctx)
    assert len(sentences) == 1


def test_choose_num_ctx_clamped(monkeypatch):
    dataset = _load_dataset(monkeypatch, "text")

    monkeypatch.setattr(dataset.random, "gauss", lambda mu, sig: 10)
    assert dataset._choose_num_ctx(5) == 5

    monkeypatch.setattr(dataset.random, "gauss", lambda mu, sig: -5)
    assert dataset._choose_num_ctx(5) == 0

    monkeypatch.setattr(dataset.random, "gauss", lambda mu, sig: 3)
    assert dataset._choose_num_ctx(2) == 2


def test_ensure_model_failure(monkeypatch):
    dataset = _load_dataset(monkeypatch, "text", tags={"models": []})

    with pytest.raises(RuntimeError):
        dataset._ensure_model()
