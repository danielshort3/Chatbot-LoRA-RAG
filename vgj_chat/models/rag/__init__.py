"""RAG model orchestration."""

import sys
import types

from . import boot as _boot_mod
from .generation import chat
from .retrieval import SentenceWindowRetriever, retrieve_unique, retrieve_windows


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
    "retrieve_windows",
    "SentenceWindowRetriever",
    "chat",
]
