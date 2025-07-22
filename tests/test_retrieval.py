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
        return DummyVector([0.0, 0.0, 0.0])


class DummyIndex:
    def search(self, vec, k):
        return None, [[0, 1, 2, 3]]


class DummyReranker:
    def predict(self, pairs):
        return [0.8, 0.5, 0.9, 0.4]


def setup_module(module):
    rag.INDEX = DummyIndex()
    rag.TEXTS = ["t1", "t2", "t3", "t4"]
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
