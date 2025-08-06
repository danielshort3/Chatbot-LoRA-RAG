from dataclasses import replace

from vgj_chat import config
from vgj_chat.models import rag


class DummyVector(list):
    def astype(self, _):
        return self

    def __getitem__(self, key):
        return [self]


class DummyEmbedder:
    def encode(self, query, normalize_embeddings=True):
        if isinstance(query, list):
            return [DummyVector([0.0, 0.0, 0.0]) for _ in query]
        return DummyVector([0.0, 0.0, 0.0])


class DummyIndex:
    def search(self, vec, k):
        return None, [[0, 1, 2, 3]]


class DummyReranker:
    def predict(self, pairs):
        return [0.8, 0.5, 0.9, 0.4]


def setup_module(module):
    rag.INDEX = DummyIndex()
    rag.TEXTS = [
        "t1 a. t1 b. t1 c. t1 d.",
        "t2 a. t2 b. t2 c. t2 d.",
        "t3 a. t3 b. t3 c. t3 d.",
        "t4 a. t4 b. t4 c. t4 d.",
    ]
    rag.URLS = ["u1", "u1", "u2", "u3"]
    rag.EMBEDDER = DummyEmbedder()
    rag.RERANKER = DummyReranker()
    rag.CHAT = None
    rag._BOOTED = True
    rag.CFG = replace(config.CFG, top_k=2, score_min=0.0)


def teardown_module(module):
    rag._BOOTED = False


def test_retrieve_unique_shape():
    results = rag.retrieve_unique("q")
    assert isinstance(results, list)
    assert len(results) == 2
    for score, text, url in results:
        assert isinstance(score, float)
        assert isinstance(text, str)
        assert isinstance(url, str)
    assert results[0][0] >= results[1][0]


def test_retrieve_windows_shape():
    results = rag.retrieve_windows("q")
    assert isinstance(results, list)
    assert len(results) <= 3
    for block in results:
        assert block.startswith("<DOC_ID:")
        assert "<URL:" in block
        assert "<DATE:" in block
        assert "<PARA_ID:" in block
