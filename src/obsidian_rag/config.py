"""Optional TOML configuration with environment-variable overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INDEX_DIR = ".obsidianrag"


@dataclass
class RagConfig:
    """Runtime defaults; every field can be overridden by a CLI flag."""

    index_dir: str = DEFAULT_INDEX_DIR
    embedder: str = "hashing"
    max_chars: int = 1000
    overlap: int = 100
    top_k: int = 5
    min_score: float = 0.0
    hashing_dim: int = 1024


_ENV_OVERRIDES = {
    "OBSIDIAN_RAG_INDEX": ("index_dir", str),
    "OBSIDIAN_RAG_EMBEDDER": ("embedder", str),
    "OBSIDIAN_RAG_TOP_K": ("top_k", int),
}


def load_config(path: str | Path | None = None) -> RagConfig:
    """Load defaults from an optional TOML file, then apply env overrides."""
    config = RagConfig()
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"config file not found: {config_path}")
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    for env_name, (field, cast) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw:
            setattr(config, field, cast(raw))
    return config
