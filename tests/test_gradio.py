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
