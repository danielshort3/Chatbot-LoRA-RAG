from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import get_type_hints

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
    # controls GPU usage for FAISS separately from the rest of the app
    faiss_cuda: bool = torch.cuda.is_available()
    debug: bool = False

    # authentication
    hf_token: str | None = None

    # UI mode
    compare_mode: bool = False

    @staticmethod
    def _convert(value: str, typ: type):
        """Convert *value* to *typ* for overrides."""
        if typ is bool:
            return value.lower() in {"1", "true", "yes", "on"}
        if typ is int:
            return int(value)
        if typ is float:
            return float(value)
        if typ is Path:
            return Path(value)
        return value

    @classmethod
    def from_env(cls) -> "Config":
        """Create config using environment variables prefixed with ``VGJ_``."""
        defaults = cls()
        updates = {}
        hints = get_type_hints(cls)
        for f in fields(cls):
            env_name = f"VGJ_{f.name.upper()}"
            if env_name in os.environ:
                typ = hints.get(f.name, f.type)
                updates[f.name] = cls._convert(os.environ[env_name], typ)
        return replace(defaults, **updates)

    def apply_cli_args(self, args: argparse.Namespace) -> "Config":
        """Return a new config overriding with parsed command-line arguments."""
        updates = {}
        hints = get_type_hints(self.__class__)
        for f in fields(self):
            val = getattr(args, f.name, None)
            if val is not None:
                typ = hints.get(f.name, f.type)
                updates[f.name] = self._convert(val, typ)
        return replace(self, **updates)

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add command-line arguments for all config fields."""
        for f in fields(cls):
            parser.add_argument(
                f"--{f.name.replace('_', '-')}",
                dest=f.name,
                type=str,
                help=f"Override {f.name} (default: %(default)s)",
                default=None,
            )


CFG = Config.from_env()
