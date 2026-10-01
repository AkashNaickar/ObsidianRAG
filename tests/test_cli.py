import json

import pytest

from obsidian_rag.cli import main


def _make_vault(tmp_path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# Apples\n\nApples are red fruit.\n", encoding="utf-8")
    (notes / "b.md").write_text(
        "# Rockets\n\nRockets burn fuel to reach orbit.\n", encoding="utf-8"
    )


def test_ingest_and_query_json(tmp_path, capsys) -> None:
    _make_vault(tmp_path)
    index = tmp_path / "idx"

    assert main(["ingest", str(tmp_path / "notes"), "--index", str(index), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunks"] >= 2
    assert payload["embedder"] == "hashing"

    assert main(["query", "red fruit", "--index", str(index), "--json", "-k", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["citations"][0]["source"] == "a.md"


def test_ingest_and_query_text(tmp_path, capsys) -> None:
    _make_vault(tmp_path)
    index = tmp_path / "idx"

    assert main(["ingest", str(tmp_path / "notes"), "--index", str(index)]) == 0
    assert "Indexed" in capsys.readouterr().out

    assert main(["query", "rocket fuel", "--index", str(index), "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "Citations:" in out
    assert "b.md" in out


def test_query_without_matches(tmp_path, capsys) -> None:
    _make_vault(tmp_path)
    index = tmp_path / "idx"
    main(["ingest", str(tmp_path / "notes"), "--index", str(index)])
    capsys.readouterr()

    assert main(["query", "zzzz", "--index", str(index), "--min-score", "0.99"]) == 0
    assert "No relevant passages" in capsys.readouterr().out


def test_stats(tmp_path, capsys) -> None:
    _make_vault(tmp_path)
    index = tmp_path / "idx"
    main(["ingest", str(tmp_path / "notes"), "--index", str(index)])
    capsys.readouterr()

    assert main(["stats", "--index", str(index)]) == 0
    assert "Embedder:" in capsys.readouterr().out

    assert main(["stats", "--index", str(index), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["chunks"] >= 2


def test_config_sets_embedder(tmp_path, capsys) -> None:
    _make_vault(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text('embedder = "tfidf"\n', encoding="utf-8")

    assert (
        main(
            [
                "--config",
                str(config),
                "ingest",
                str(tmp_path / "notes"),
                "--index",
                str(tmp_path / "idx"),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["embedder"] == "tfidf"


def test_missing_index_is_reported(tmp_path, capsys) -> None:
    assert main(["query", "x", "--index", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_notes_directory_is_reported(tmp_path, capsys) -> None:
    assert main(["ingest", str(tmp_path / "missing"), "--index", str(tmp_path / "idx")]) == 1
    assert "error:" in capsys.readouterr().err


def test_tfidf_index_is_reloadable_for_queries(tmp_path, capsys) -> None:
    _make_vault(tmp_path)
    index = tmp_path / "idx"

    assert (
        main(["ingest", str(tmp_path / "notes"), "--index", str(index), "--embedder", "tfidf"]) == 0
    )
    capsys.readouterr()

    assert main(["query", "rocket fuel", "--index", str(index), "--json", "-k", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["citations"][0]["source"] == "b.md"


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "obsidian-rag" in capsys.readouterr().out
