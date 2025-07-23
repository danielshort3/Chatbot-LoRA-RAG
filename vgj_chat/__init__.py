"""VGJ Chat package."""

from .config import CFG
from .models import rag


def chat(question: str) -> str:
    """Return an answer to *question* using the RAG model."""
    return rag.chat(question)


def run_enhanced(question: str) -> str:
    """Return an answer using the enhanced RAG pipeline."""
    return rag.run_enhanced(question)


def run_baseline(question: str) -> str:
    """Return a baseline answer without retrieval or LoRA."""
    return rag.run_baseline(question)


__all__ = ["CFG", "chat", "run_enhanced", "run_baseline"]
