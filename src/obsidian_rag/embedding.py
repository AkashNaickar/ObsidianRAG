"""Embedding backends.

Two deterministic, fully offline embedders are always available:

* :class:`HashingEmbedder` - signed feature hashing over word n-grams. No
  training step, no vocabulary to persist.
* :class:`TfidfEmbedder` - classic TF-IDF fitted on the ingested corpus.

:class:`SentenceTransformerEmbedder` is optional and only imported when
``sentence-transformers`` is installed (``pip install 'obsidian-rag[embeddings]'``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from obsidian_rag.text import ngrams, tokenize


class Embedder(Protocol):
    """Interface implemented by all embedding backends."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def state(self) -> dict[str, Any]: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic signed feature hashing over word unigrams..n-grams."""

    name = "hashing"

    def __init__(self, dim: int = 1024, ngram_range: tuple[int, int] = (1, 2)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        low, high = ngram_range
        if low <= 0 or high < low:
            raise ValueError("ngram_range must satisfy 0 < low <= high")
        self.dim = int(dim)
        self.ngram_range = (int(low), int(high))

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if value & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            counts: dict[str, int] = {}
            for size in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for gram in ngrams(tokens, size):
                    counts[gram] = counts.get(gram, 0) + 1
            for gram, count in counts.items():
                index, sign = self._hash(gram)
                matrix[row, index] += sign * (1.0 + math.log(count))
        return _normalize(matrix)

    def state(self) -> dict[str, Any]:
        return {"dim": self.dim, "ngram_range": list(self.ngram_range)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> HashingEmbedder:
        return cls(dim=int(state["dim"]), ngram_range=tuple(state["ngram_range"]))


class TfidfEmbedder:
    """TF-IDF vectors with a vocabulary fitted on the ingested corpus."""

    name = "tfidf"

    def __init__(self, max_features: int = 50_000, min_df: int = 1) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if min_df <= 0:
            raise ValueError("min_df must be positive")
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self.dim = 0

    @property
    def fitted(self) -> bool:
        return bool(self._vocab)

    def fit(self, corpus: Sequence[str]) -> TfidfEmbedder:
        document_frequency: dict[str, int] = {}
        for text in corpus:
            for token in set(tokenize(text)):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        terms = [term for term, df in document_frequency.items() if df >= self.min_df]
        terms.sort(key=lambda term: (-document_frequency[term], term))
        terms = terms[: self.max_features]
        terms.sort()

        self._vocab = {term: index for index, term in enumerate(terms)}
        total = len(corpus)
        idf = np.zeros(len(terms), dtype=np.float32)
        for term, index in self._vocab.items():
            idf[index] = math.log((1.0 + total) / (1.0 + document_frequency[term])) + 1.0
        self._idf = idf
        self.dim = len(terms)
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted or self._idf is None:
            raise RuntimeError("TfidfEmbedder must be fitted before embedding")
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                index = self._vocab.get(token)
                if index is not None:
                    matrix[row, index] = (1.0 + math.log(count)) * float(self._idf[index])
        return _normalize(matrix)

    def state(self) -> dict[str, Any]:
        if not self.fitted or self._idf is None:
            raise RuntimeError("TfidfEmbedder must be fitted before serialising")
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "vocab": list(self._vocab),
            "idf": self._idf.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TfidfEmbedder:
        embedder = cls(
            max_features=int(state["max_features"]),
            min_df=int(state["min_df"]),
        )
        vocab = [str(term) for term in state["vocab"]]
        embedder._vocab = {term: index for index, term in enumerate(vocab)}
        embedder._idf = np.asarray(state["idf"], dtype=np.float32)
        embedder.dim = len(vocab)
        return embedder


class SentenceTransformerEmbedder:
    """Optional dense embedder backed by ``sentence-transformers``."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install 'obsidian-rag[embeddings]'"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def state(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> SentenceTransformerEmbedder:
        return cls(model_name=str(state["model_name"]))


_BUILDERS: dict[str, type] = {
    "hashing": HashingEmbedder,
    "tfidf": TfidfEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}


def build_embedder(name: str = "hashing", **kwargs: Any) -> Embedder:
    """Construct an embedder by name."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        known = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"unknown embedder {name!r}; expected one of: {known}") from None
    return builder(**kwargs)  # type: ignore[no-any-return]
