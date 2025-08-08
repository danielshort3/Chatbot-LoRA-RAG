"""VGJ Chat package."""

from .config import CFG
from .models import rag


def chat(question: str):
    """Return an answer to *question* using the RAG model."""
    return rag.chat(question)


__all__ = ["CFG", "chat"]
