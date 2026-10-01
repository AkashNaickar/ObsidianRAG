"""ObsidianRAG: minimal, fully offline retrieval-augmented search over Markdown notes.

The package deliberately has a tiny dependency surface (``numpy`` only) so the
default pipeline runs without network access, API keys, or model downloads.
"""

from __future__ import annotations

from obsidian_rag.chunking import Chunk, chunk_markdown
from obsidian_rag.embedding import HashingEmbedder, TfidfEmbedder, build_embedder
from obsidian_rag.rag import RagError, RagIndex
from obsidian_rag.store import SearchHit, VectorStore

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "HashingEmbedder",
    "RagError",
    "RagIndex",
    "SearchHit",
    "TfidfEmbedder",
    "VectorStore",
    "__version__",
    "build_embedder",
    "chunk_markdown",
]
