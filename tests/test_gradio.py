import time
from contextlib import suppress

from vgj_chat.models import rag
from vgj_chat.ui import gradio_app


def test_gradio_launch(monkeypatch):
    monkeypatch.setattr(rag, "_ensure_boot", lambda: None)
    monkeypatch.setattr(gradio_app, "_ensure_boot", lambda: None)
    demo = gradio_app.build_demo()
    assert hasattr(demo, "launch")
    with suppress(Exception):
        demo.launch(prevent_thread_lock=True)
        time.sleep(0.5)
        demo.close()


def test_build_demo_compare_mode(monkeypatch):
    from dataclasses import replace
    from vgj_chat import config

    monkeypatch.setattr(rag, "_ensure_boot", lambda: None)
    monkeypatch.setattr(gradio_app, "_ensure_boot", lambda: None)

    created: list[object] = []

    class CountingComponent:
        def __init__(self, *a, **k):
            created.append(self)

        def submit(self, *a, **k):
            return self

        def then(self, *a, **k):
            return self

    monkeypatch.setattr(gradio_app.gr, "Chatbot", CountingComponent)
    monkeypatch.setattr(
        gradio_app.gr,
        "Row",
        lambda *a, **k: gradio_app.gr.Blocks(),
        raising=False,
    )

    original_cfg = config.CFG
    new_cfg = replace(config.CFG, compare_mode=True)
    config.CFG = gradio_app.CFG = new_cfg

    try:
        gradio_app.build_demo()
    finally:
        config.CFG = gradio_app.CFG = original_cfg

    assert len(created) == 2
