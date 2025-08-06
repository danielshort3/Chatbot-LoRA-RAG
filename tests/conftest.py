import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Minimal stubs for heavy dependencies
if "torch" not in sys.modules:
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    torch_stub.ones_like = lambda x: [1] * len(x)
    sys.modules["torch"] = torch_stub
else:  # pragma: no cover - ensure stub has ones_like
    if not hasattr(sys.modules["torch"], "ones_like"):
        sys.modules["torch"].ones_like = lambda x: [1] * len(x)

for mod_name in ["faiss", "sentence_transformers", "transformers", "peft"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = ModuleType(mod_name)

# faiss helpers used in code
sys.modules["faiss"].read_index = lambda *a, **k: None

sys.modules["sentence_transformers"].CrossEncoder = object
sys.modules["sentence_transformers"].SentenceTransformer = object

sys.modules["transformers"].AutoModelForCausalLM = object
sys.modules["transformers"].AutoTokenizer = object
sys.modules["transformers"].BitsAndBytesConfig = object
sys.modules["transformers"].TextIteratorStreamer = object
sys.modules["transformers"].pipeline = lambda *a, **k: None

sys.modules["peft"].PeftModel = object


# ------------------------------------------------------------------
# Deterministic retrieval helpers
# ------------------------------------------------------------------


class FixedEmbedder:
    """Simple keyword based embedder for tests."""

    def encode(self, texts: Sequence[str] | str, normalize_embeddings: bool = True):
        def _vec(txt: str) -> np.ndarray:
            v = np.zeros(3, dtype="float32")
            for i, token in enumerate(("alpha", "beta", "gamma")):
                if token in txt.lower():
                    v[i] = 1.0
            return v

        if isinstance(texts, list):
            return [_vec(t) for t in texts]
        return _vec(texts)


class FakeIndex:
    """FAISS-like index performing brute-force dot search."""

    def __init__(self, texts: Sequence[str], embedder: FixedEmbedder):
        self.vecs = [embedder.encode(t) for t in texts]

    def search(self, q_vec, k: int):
        sims = [float(np.dot(v, q_vec[0])) for v in self.vecs]
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return None, [order[:k]]


@pytest.fixture
def set_retrieval_env(monkeypatch):
    """Configure :mod:`vgj_chat.models.rag.boot` with deterministic resources."""

    from vgj_chat.models.rag import boot as _boot

    def _set(texts: Sequence[str], urls: Sequence[str] | None = None):
        if urls is None:
            urls = [f"u{i}" for i in range(len(texts))]
        embedder = FixedEmbedder()
        index = FakeIndex(texts, embedder)
        monkeypatch.setattr(_boot, "TEXTS", list(texts))
        monkeypatch.setattr(_boot, "URLS", list(urls))
        monkeypatch.setattr(_boot, "EMBEDDER", embedder)
        monkeypatch.setattr(_boot, "INDEX", index)
        monkeypatch.setattr(_boot, "_RETRIEVAL_DISABLED", False)
        monkeypatch.setattr(_boot, "_ensure_boot", lambda: None)
        monkeypatch.setattr(_boot, "_BOOTED", True)
        return embedder, index

    return _set
