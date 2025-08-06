"""RAG model orchestration."""

import sys
import types

from . import baseline as _baseline
from . import boot as _boot_mod
from .generation import answer_stream, chat, run_enhanced
from .retrieval import retrieve_unique

_baseline_mode = _baseline._baseline_mode


def run_baseline(question: str) -> str:
    with _baseline_mode():
        assert _boot_mod.CHAT
        generated = _boot_mod.CHAT(
            question,
            do_sample=True,
            temperature=0.8,
            max_new_tokens=_boot_mod.CFG.max_new_tokens,
        )[0]["generated_text"]
    return generated.strip()


class _RagModule(types.ModuleType):
    """Module type that proxies unknown attributes to :mod:`.boot`."""

    def __getattr__(self, name):  # pragma: no cover - simple delegation
        return getattr(_boot_mod, name)

    def __setattr__(self, name, value):  # pragma: no cover - simple delegation
        if hasattr(_boot_mod, name):
            setattr(_boot_mod, name, value)
        else:
            super().__setattr__(name, value)


# Replace current module to delegate attribute access to :mod:`.boot`
sys.modules[__name__].__class__ = _RagModule  # type: ignore[misc]

__all__ = [
    "retrieve_unique",
    "answer_stream",
    "chat",
    "run_enhanced",
    "run_baseline",
    "_baseline_mode",
]
