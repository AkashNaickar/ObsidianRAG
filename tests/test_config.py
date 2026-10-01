import pytest

from obsidian_rag.config import RagConfig, load_config


def test_defaults() -> None:
    config = RagConfig()
    assert config.embedder == "hashing"
    assert config.top_k == 5


def test_load_from_toml(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'index_dir = "custom"\nembedder = "tfidf"\ntop_k = 3\nmax_chars = 500\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.index_dir == "custom"
    assert config.embedder == "tfidf"
    assert config.top_k == 3
    assert config.max_chars == 500


def test_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OBSIDIAN_RAG_EMBEDDER", "tfidf")
    monkeypatch.setenv("OBSIDIAN_RAG_TOP_K", "7")
    monkeypatch.setenv("OBSIDIAN_RAG_INDEX", "env-index")
    config = load_config()
    assert config.embedder == "tfidf"
    assert config.top_k == 7
    assert config.index_dir == "env-index"


def test_missing_config_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
