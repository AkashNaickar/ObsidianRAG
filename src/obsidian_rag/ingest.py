"""Discovery and chunking of Markdown note folders."""

from __future__ import annotations

from pathlib import Path

from obsidian_rag.chunking import Chunk, chunk_markdown

MARKDOWN_SUFFIXES = {".md", ".markdown"}
SKIP_DIRS = {".git", ".obsidian", ".trash", "node_modules", "__pycache__", ".obsidianrag"}


def discover_notes(root: str | Path) -> list[Path]:
    """Return sorted Markdown files under ``root``, skipping tooling folders."""
    directory = Path(root)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    notes: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        relative_parts = path.relative_to(directory).parts[:-1]
        if any(part.startswith(".") or part in SKIP_DIRS for part in relative_parts):
            continue
        if path.name.startswith("."):
            continue
        notes.append(path)
    return notes


def chunk_notes(
    root: str | Path,
    *,
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    """Read every Markdown note under ``root`` and chunk it."""
    directory = Path(root)
    chunks: list[Chunk] = []
    for path in discover_notes(directory):
        text = path.read_text(encoding="utf-8", errors="replace")
        source = path.relative_to(directory).as_posix()
        chunks.extend(chunk_markdown(text, source, max_chars=max_chars, overlap=overlap))
    return chunks
