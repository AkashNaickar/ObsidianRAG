from pathlib import Path

from obsidian_rag.rag import RagIndex

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "notes"


def test_example_vault_roundtrip(tmp_path) -> None:
    index = RagIndex.build(EXAMPLES)
    assert index.store.size > 0

    result = index.answer("paragraph packing and overlap for long blocks", top_k=3)
    assert result["citations"]
    assert result["citations"][0]["source"] == "chunking.md"

    index.save(tmp_path / "idx")
    loaded = RagIndex.load(tmp_path / "idx")
    hits = loaded.query("embeddings vector similarity", top_k=1)
    assert hits[0].chunk.source == "embeddings.md"
