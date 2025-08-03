from dataclasses import replace

import pytest
from vgj_chat.models import rag


def test_boot_raises_when_model_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "load_index", lambda *_: None)
    monkeypatch.setattr(rag, "load_metadata", lambda *_: ([], []))
    monkeypatch.setattr(rag, "SentenceTransformer", lambda *_, **__: None)
    monkeypatch.setattr(rag, "CrossEncoder", lambda *_, **__: None)
    monkeypatch.setattr(rag.torch, "float16", object(), raising=False)
    monkeypatch.setattr(rag, "BitsAndBytesConfig", lambda *_, **__: None)
    missing = tmp_path / "missing"
    monkeypatch.setattr(rag, "CFG", replace(rag.CFG, merged_model_dir=missing))
    with pytest.raises(FileNotFoundError) as exc:
        rag._boot()
    assert str(missing) in str(exc.value)
