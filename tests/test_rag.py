import json

import pytest

from obsidian_rag.rag import RagError, RagIndex, build_answer

NOTES = {
    "alpha.md": (
        "# Apples\n\n"
        "Apples are red fruit that grow on trees.\n\n"
        "## Varieties\n\n"
        "Fuji and Gala are sweet apples.\n"
    ),
    "beta.md": (
        "# Rockets\n\nRockets burn fuel to reach orbit. Thrust and delta-v matter a lot.\n"
    ),
}


@pytest.fixture
def vault(tmp_path):
    for name, text in NOTES.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_build_and_query(vault) -> None:
    index = RagIndex.build(vault)
    hits = index.query("sweet red fruit", top_k=3)
    assert hits
    assert hits[0].chunk.source == "alpha.md"
    assert hits[0].score > 0


def test_tfidf_backend(vault) -> None:
    index = RagIndex.build(vault, embedder="tfidf")
    assert index.embedder.name == "tfidf"
    hits = index.query("rocket fuel orbit", top_k=1)
    assert hits[0].chunk.source == "beta.md"


def test_save_load_roundtrip(vault, tmp_path) -> None:
    index = RagIndex.build(vault, embedder="tfidf")
    out = tmp_path / "idx"
    manifest = index.save(out)
    assert manifest["num_chunks"] == index.store.size

    loaded = RagIndex.load(out)
    assert loaded.stats()["embedder"] == "tfidf"
    hits = loaded.query("rocket fuel orbit", top_k=1)
    assert hits[0].chunk.source == "beta.md"


def test_answer_contains_citations(vault) -> None:
    index = RagIndex.build(vault)
    result = index.answer("rocket fuel", top_k=2)
    assert result["citations"]
    assert result["citations"][0]["source"] == "beta.md"
    assert "[1]" in result["answer"]


def test_empty_directory_raises(tmp_path) -> None:
    with pytest.raises(RagError):
        RagIndex.build(tmp_path)


def test_stats(vault) -> None:
    index = RagIndex.build(vault)
    stats = index.stats()
    assert stats["chunks"] >= 2
    assert set(stats["sources"]) == {"alpha.md", "beta.md"}
    assert stats["embedder"] == "hashing"


def test_configurable_hashing_dim(vault) -> None:
    index = RagIndex.build(vault, embedder="hashing", dim=64)
    assert index.store.dim == 64


def test_build_answer_without_hits() -> None:
    result = build_answer("question", [])
    assert result["answer"] == "No relevant passages found."
    assert result["citations"] == []


def test_load_rejects_invalid_manifest(vault, tmp_path) -> None:
    out = tmp_path / "idx"
    RagIndex.build(vault).save(out)
    (out / "manifest.json").write_text(json.dumps({"embedder": "nope"}), encoding="utf-8")
    with pytest.raises(RagError):
        RagIndex.load(out)


def test_load_rejects_unknown_embedder(vault, tmp_path) -> None:
    out = tmp_path / "idx"
    RagIndex.build(vault).save(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["embedder"]["name"] = "nope"
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RagError):
        RagIndex.load(out)
