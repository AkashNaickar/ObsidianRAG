import numpy as np
import pytest

from obsidian_rag.chunking import Chunk
from obsidian_rag.store import SearchHit, VectorStore


def _chunk(index: int, text: str = "hello") -> Chunk:
    return Chunk(
        chunk_id=f"c{index}",
        source="a.md",
        heading="H",
        text=text,
        start_line=index,
        end_line=index,
    )


def test_save_load_roundtrip(tmp_path) -> None:
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
    store = VectorStore(vectors, [_chunk(1), _chunk(2)])
    manifest = store.save(tmp_path, {"embedder": {"name": "hashing", "state": {"dim": 2}}})
    assert manifest["num_chunks"] == 2

    loaded, loaded_manifest = VectorStore.load(tmp_path)
    assert loaded.size == 2
    assert loaded.dim == 2
    np.testing.assert_allclose(loaded.vectors, vectors)
    assert loaded.chunks[0].chunk_id == "c1"
    assert loaded_manifest["embedder"]["name"] == "hashing"


def test_search_ordering_and_min_score() -> None:
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
    store = VectorStore(vectors, [_chunk(1, "x"), _chunk(2, "y")])
    hits = store.search(np.array([0.9, 0.1]), top_k=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["c1", "c2"]
    assert hits[0].score > hits[1].score

    filtered = store.search(np.array([1, 0]), top_k=2, min_score=0.5)
    assert len(filtered) == 1
    assert filtered[0].chunk.chunk_id == "c1"


def test_search_validates_inputs() -> None:
    store = VectorStore(np.array([[1, 0]], dtype=np.float32), [_chunk(1)])
    with pytest.raises(ValueError):
        store.search(np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        store.search(np.array([1, 0]), top_k=0)


def test_row_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        VectorStore(np.array([[1, 0]]), [])


def test_non_2d_vectors_raise() -> None:
    with pytest.raises(ValueError):
        VectorStore(np.array([1, 0], dtype=np.float32), [_chunk(1)])


def test_empty_store_search() -> None:
    store = VectorStore(np.zeros((0, 0), dtype=np.float32), [])
    assert store.size == 0
    assert store.dim == 0
    assert store.search(np.zeros(0)) == []


def test_load_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path)


def test_hit_to_dict() -> None:
    hit = SearchHit(0.5, _chunk(3))
    payload = hit.to_dict()
    assert payload["score"] == 0.5
    assert payload["chunk_id"] == "c3"
