from contextlib import contextmanager

from vgj_chat.models import rag


def test_run_baseline_disables_retrieval(monkeypatch):
    events = []

    @contextmanager
    def fake_mode():
        events.append(("enter", rag._RETRIEVAL_DISABLED))
        rag._RETRIEVAL_DISABLED = True
        try:
            yield
        finally:
            events.append(("exit", rag._RETRIEVAL_DISABLED))
            rag._RETRIEVAL_DISABLED = False

    monkeypatch.setattr(rag, "_baseline_mode", fake_mode)
    monkeypatch.setattr(rag, "CHAT", lambda *a, **k: [{"generated_text": "resp"}])

    result = rag.run_baseline("hello")

    assert result == "resp"
    assert events == [("enter", False), ("exit", True)]
    assert rag._RETRIEVAL_DISABLED is False
