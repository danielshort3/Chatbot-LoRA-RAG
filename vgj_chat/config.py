from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass(frozen=True)
class Config:
    """Application configuration."""

    # paths
    index_path: Path = Path("faiss.index")
    meta_path: Path = Path("meta.jsonl")
    lora_dir: Path = Path("lora-vgj-checkpoint")

    # models
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "BAAI/bge-reranker-base"

    # RAG settings
    top_k: int = 5
    score_min: float = 0.0
    max_new_tokens: int = 512

    # similarity guard-rail
    sim_threshold: float = 0.80

    # misc
    cuda: bool = torch.cuda.is_available()
    debug: bool = True


CFG = Config()
