import importlib
import re
import sys
import types

from types import ModuleType


def _load_dataset(monkeypatch, text: str):
    """Reload dataset with heavy dependencies stubbed and return module and stubs."""

    class _DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub = types.SimpleNamespace(
        no_grad=lambda: _DummyNoGrad(),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "trafilatura", ModuleType("trafilatura"))

    dataset = importlib.reload(importlib.import_module("vgj_chat.data.dataset"))

    class DummyTensor(list):
        @property
        def shape(self):
            return (len(self),)

    class DummyInput(dict):
        def __init__(self):
            super().__init__({"input_ids": DummyTensor([1, 2, 3])})
            self.input_ids = self["input_ids"]

        def to(self, device):
            return self

    class DummyTokenizer:
        def __init__(self, text: str):
            self.text = text
            self.eos_token_id = 0

        def __call__(self, prompt, return_tensors=None):
            return DummyInput()

        def decode(self, ids, skip_special_tokens=True):
            return self.text

    class DummyLLM:
        device = "cpu"

        def generate(self, **kwargs):
            return [DummyTensor([0, 1, 2, 3, 4])]

    tok = DummyTokenizer(text)
    llm = DummyLLM()
    return dataset, tok, llm


def test_gen_context_returns_complete_sentences(monkeypatch):
    incomplete_text = (
        "The first sentence is complete. "
        "This is another complete sentence. "
        "This final sentence is cut off"
    )
    dataset, tok, llm = _load_dataset(monkeypatch, incomplete_text)

    ctx = dataset._gen_context("question", "answer", tok, llm)

    assert ctx.endswith((".", "!", "?")), "Context should end with punctuation"
    sentences = re.findall(r"[^.!?]+[.!?]", ctx)
    assert 2 <= len(sentences) <= 3
    reconstructed = " ".join(s.strip() for s in sentences)
    assert ctx == reconstructed
    assert "cut off" not in ctx


def test_gen_context_single_sentence(monkeypatch):
    single_sentence = "Only one complete sentence here."
    dataset, tok, llm = _load_dataset(monkeypatch, single_sentence)

    ctx = dataset._gen_context("question", "answer", tok, llm)

    assert ctx.endswith((".", "!", "?"))
    assert ctx == single_sentence
    sentences = re.findall(r"[^.!?]+[.!?]", ctx)
    assert len(sentences) == 1


def test_choose_num_ctx_clamped(monkeypatch):
    dataset, _, _ = _load_dataset(monkeypatch, "text")

    monkeypatch.setattr(dataset.random, "gauss", lambda mu, sig: 10)
    assert dataset._choose_num_ctx(5) == 5

    monkeypatch.setattr(dataset.random, "gauss", lambda mu, sig: -5)
    assert dataset._choose_num_ctx(5) == 0

    monkeypatch.setattr(dataset.random, "gauss", lambda mu, sig: 3)
    assert dataset._choose_num_ctx(2) == 2

