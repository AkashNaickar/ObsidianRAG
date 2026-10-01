"""High-level build / save / load / query pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obsidian_rag.embedding import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    build_embedder,
)
from obsidian_rag.ingest import chunk_notes
from obsidian_rag.store import SearchHit, VectorStore


class RagError(RuntimeError):
    """Raised for user-facing pipeline errors."""


def _embedding_text(heading: str, text: str) -> str:
    return f"{heading}\n{text}" if heading else text


def _embedder_from_state(name: str, state: dict[str, Any]) -> Embedder:
    if name == "hashing":
        return HashingEmbedder.from_state(state)
    if name == "tfidf":
        return TfidfEmbedder.from_state(state)
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder.from_state(state)
    raise RagError(f"index was built with unsupported embedder {name!r}")


def build_answer(question: str, hits: list[SearchHit]) -> dict[str, Any]:
    """Turn scored hits into a deterministic extractive answer with citations."""
    citations: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        citations.append(
            {
                "index": rank,
                "source": chunk.source,
                "heading": chunk.heading,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": round(hit.score, 6),
            }
        )
        passages.append({"index": rank, "text": chunk.text})

    if passages:
        answer = "\n\n".join(f"[{item['index']}] {item['text']}" for item in passages)
    else:
        answer = "No relevant passages found."

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "passages": passages,
    }


@dataclass
class RagIndex:
    """A loaded or freshly built index."""

    embedder: Embedder
    store: VectorStore
    source_root: str = ""

    @classmethod
    def build(
        cls,
        notes_dir: str | Path,
        *,
        embedder: str = "hashing",
        max_chars: int = 1000,
        overlap: int = 100,
        **embedder_kwargs: Any,
    ) -> RagIndex:
        chunks = chunk_notes(notes_dir, max_chars=max_chars, overlap=overlap)
        if not chunks:
            raise RagError(f"no Markdown notes found under {Path(notes_dir)}")

        backend = build_embedder(embedder, **embedder_kwargs)
        texts = [_embedding_text(chunk.heading, chunk.text) for chunk in chunks]
        if backend.name == "tfidf":
            backend.fit(texts)  # type: ignore[attr-defined]
        vectors = backend.embed(texts)
        return cls(embedder=backend, store=VectorStore(vectors, chunks), source_root=str(notes_dir))

    def save(self, index_dir: str | Path) -> dict[str, Any]:
        manifest = {
            "embedder": {"name": self.embedder.name, "state": self.embedder.state()},
            "source_root": self.source_root,
            "sources": sorted({chunk.source for chunk in self.store.chunks}),
        }
        return self.store.save(index_dir, manifest)

    @classmethod
    def load(cls, index_dir: str | Path) -> RagIndex:
        store, manifest = VectorStore.load(index_dir)
        embedder_info = manifest.get("embedder")
        if not isinstance(embedder_info, dict):
            raise RagError(f"{index_dir} does not contain a valid embedder manifest")
        backend = _embedder_from_state(str(embedder_info["name"]), embedder_info["state"])
        return cls(embedder=backend, store=store, source_root=str(manifest.get("source_root", "")))

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, top_k=top_k, min_score=min_score)

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        return build_answer(question, self.query(question, top_k=top_k, min_score=min_score))

    def stats(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.name,
            "dim": self.store.dim,
            "chunks": self.store.size,
            "sources": sorted({chunk.source for chunk in self.store.chunks}),
            "source_root": self.source_root,
        }
