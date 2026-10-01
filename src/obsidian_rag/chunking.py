"""Markdown-aware chunking that preserves heading and source-line provenance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage of a Markdown note."""

    chunk_id: str
    source: str
    heading: str
    text: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "heading": self.heading,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Chunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            source=str(data["source"]),
            heading=str(data["heading"]),
            text=str(data["text"]),
            start_line=int(data["start_line"]),  # type: ignore[arg-type]
            end_line=int(data["end_line"]),  # type: ignore[arg-type]
        )


@dataclass
class _Unit:
    """A paragraph-sized block under a single heading."""

    heading: str
    start_line: int
    end_line: int
    text: str


def _split_frontmatter(text: str) -> tuple[list[str], int]:
    """Return body lines and the number of leading lines that were metadata."""
    lines = text.splitlines()
    removed = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                removed = index + 1
                while removed < len(lines) and not lines[removed].strip():
                    removed += 1
                break
    return lines[removed:], removed


def _collect_units(body_lines: list[str], offset: int) -> list[_Unit]:
    units: list[_Unit] = []
    heading_path: list[str] = []
    buffer: list[str] = []
    buffer_start: int | None = None
    buffer_end: int | None = None

    def flush() -> None:
        nonlocal buffer, buffer_start, buffer_end
        if buffer and buffer_start is not None and buffer_end is not None:
            units.append(
                _Unit(
                    heading=" > ".join(heading_path),
                    start_line=buffer_start,
                    end_line=buffer_end,
                    text="\n".join(buffer),
                )
            )
        buffer = []
        buffer_start = None
        buffer_end = None

    for index, raw in enumerate(body_lines):
        line_no = offset + index + 1
        match = _HEADING_RE.match(raw)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip() or "(untitled)"
            heading_path = [*heading_path[: level - 1], title]
            continue
        if raw.strip():
            if buffer_start is None:
                buffer_start = line_no
            buffer.append(raw)
            buffer_end = line_no
        else:
            flush()
    flush()
    return units


def _split_long(
    text: str, start_line: int, max_chars: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Split an oversized block into overlapping windows with line provenance."""
    if len(text) <= max_chars:
        return [(text, start_line, start_line + text.count("\n"))]

    pieces: list[tuple[str, int, int]] = []
    position = 0
    length = len(text)
    while position < length:
        end = min(position + max_chars, length)
        if end < length:
            space = text.rfind(" ", position + max_chars // 2, end)
            if space != -1:
                end = space
        piece = text[position:end].strip("\n")
        line_offset = text.count("\n", 0, position)
        piece_start = start_line + line_offset
        pieces.append((piece, piece_start, piece_start + piece.count("\n")))
        if end >= length:
            break
        position = max(end - overlap, position + 1)
    return pieces


def _finish_chunk(source: str, heading: str, text: str, start: int, end: int) -> Chunk:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return Chunk(
        chunk_id=f"{source}:{start}-{end}:{digest}",
        source=source,
        heading=heading,
        text=text,
        start_line=start,
        end_line=end,
    )


def chunk_markdown(
    text: str,
    source: str,
    *,
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    """Chunk one Markdown document.

    Paragraphs are packed up to ``max_chars`` while never crossing a heading
    boundary. Blocks longer than ``max_chars`` are split into overlapping
    windows of at most ``max_chars`` characters.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    body_lines, offset = _split_frontmatter(text)
    units = _collect_units(body_lines, offset)

    chunks: list[Chunk] = []
    heading = ""
    start = 0
    end = 0
    parts: list[str] = []

    def flush() -> None:
        nonlocal parts, start, end
        if parts:
            chunks.append(_finish_chunk(source, heading, "\n\n".join(parts), start, end))
        parts = []
        start = 0
        end = 0

    for unit in units:
        for piece, piece_start, piece_end in _split_long(
            unit.text, unit.start_line, max_chars, overlap
        ):
            if not piece.strip():
                continue
            merged_size = len("\n\n".join([*parts, piece]))
            if parts and (unit.heading != heading or merged_size > max_chars):
                flush()
            if not parts:
                heading = unit.heading
                start = piece_start
            parts.append(piece)
            end = piece_end
    flush()
    return chunks
