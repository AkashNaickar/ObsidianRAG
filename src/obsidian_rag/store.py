"""Persistent, dependency-light vector store (numpy + JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from obsidian_rag.chunking import Chunk

FORMAT_VERSION = 1
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A scored chunk returned by :meth:`VectorStore.search`."""

    score: float
    chunk: Chunk

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 6), **self.chunk.to_dict()}


class VectorStore:
    """Cosine-similarity store over pre-normalised embeddings."""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.size == 0:
            vectors = np.zeros((len(chunks), 0), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"vectors rows ({vectors.shape[0]}) must match number of chunks ({len(chunks)})"
            )
        self.vectors = vectors
        self.chunks = list(chunks)

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def save(self, index_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
        directory = Path(index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self.vectors)
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        payload = {
            **manifest,
            "format_version": FORMAT_VERSION,
            "num_chunks": self.size,
            "dim": self.dim,
        }
        (directory / MANIFEST_FILE).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload

    @classmethod
    def load(cls, index_dir: str | Path) -> tuple[VectorStore, dict[str, Any]]:
        directory = Path(index_dir)
        vectors_path = directory / VECTORS_FILE
        chunks_path = directory / CHUNKS_FILE
        manifest_path = directory / MANIFEST_FILE
        for path in (vectors_path, chunks_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found; run 'obsidian-rag ingest' first or pass --index"
                )
        vectors = np.load(vectors_path)
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(vectors, chunks), manifest

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.chunks or self.dim == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query vector has dim {query.shape[0]}, store expects {self.dim}")
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        scores = self.vectors @ query
        order = np.argsort(-scores, kind="stable")
        hits: list[SearchHit] = []
        for index in order:
            score = float(scores[int(index)])
            if score < min_score:
                continue
            hits.append(SearchHit(score, self.chunks[int(index)]))
            if len(hits) >= top_k:
                break
        return hits
