import pytest

from obsidian_rag.ingest import chunk_notes, discover_notes


def _write(path, text="# Title\n\nbody text\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_discover_notes_finds_markdown_sorted(tmp_path) -> None:
    _write(tmp_path / "b.md")
    _write(tmp_path / "a.markdown")
    (tmp_path / "notes.txt").write_text("not markdown", encoding="utf-8")

    found = [path.name for path in discover_notes(tmp_path)]
    assert found == ["a.markdown", "b.md"]


def test_discover_notes_skips_tooling_and_hidden_paths(tmp_path) -> None:
    _write(tmp_path / "keep.md")
    _write(tmp_path / ".obsidian" / "workspace.md")
    _write(tmp_path / ".git" / "COMMIT_EDITMSG.md")
    _write(tmp_path / "node_modules" / "pkg" / "readme.md")
    _write(tmp_path / "sub" / ".hidden" / "secret.md")
    _write(tmp_path / ".draft.md")

    assert [path.name for path in discover_notes(tmp_path)] == ["keep.md"]


def test_discover_notes_rejects_non_directory(tmp_path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(NotADirectoryError):
        discover_notes(missing)


def test_chunk_notes_uses_relative_posix_sources(tmp_path) -> None:
    _write(tmp_path / "sub" / "note.md", "# Heading\n\ntext\n")
    chunks = chunk_notes(tmp_path)
    assert chunks
    assert {chunk.source for chunk in chunks} == {"sub/note.md"}


def test_chunk_notes_on_empty_folder_is_empty(tmp_path) -> None:
    assert chunk_notes(tmp_path) == []
