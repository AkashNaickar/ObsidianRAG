from itertools import pairwise

import pytest

from obsidian_rag.chunking import Chunk, chunk_markdown

SAMPLE = """---
title: Demo
---

# Alpha

First paragraph about apples.

Second paragraph about oranges.

## Beta

Beta paragraph mentioning bananas.
"""


def test_headings_are_captured() -> None:
    chunks = chunk_markdown(SAMPLE, "demo.md")
    headings = {chunk.heading for chunk in chunks}
    assert "Alpha" in headings
    assert "Alpha > Beta" in headings


def test_frontmatter_excluded_and_lines_offset() -> None:
    chunks = chunk_markdown(SAMPLE, "demo.md")
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "title: Demo" not in joined
    first = chunks[0]
    assert "# Alpha" not in first.text
    assert first.start_line == 7
    assert first.end_line == 9


def test_chunk_ids_are_deterministic_and_unique() -> None:
    first = chunk_markdown(SAMPLE, "demo.md")
    second = chunk_markdown(SAMPLE, "demo.md")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_long_block_is_split_with_overlap() -> None:
    body = " ".join(f"word{index}" for index in range(400))
    text = f"# Title\n\n{body}\n"
    chunks = chunk_markdown(text, "long.md", max_chars=200, overlap=40)
    assert len(chunks) >= 3
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    for previous, current in pairwise(chunks):
        shared = set(previous.text.split()) & set(current.text.split())
        assert shared


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_markdown("x", "x.md", max_chars=0)
    with pytest.raises(ValueError):
        chunk_markdown("x", "x.md", max_chars=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_markdown("x", "x.md", max_chars=10, overlap=-1)


def test_chunk_dict_roundtrip() -> None:
    chunk = chunk_markdown(SAMPLE, "demo.md")[0]
    assert Chunk.from_dict(chunk.to_dict()) == chunk


def test_unterminated_frontmatter_is_kept_as_content() -> None:
    text = "---\ntitle: Demo\n\n# Real heading\n\nbody\n"
    chunks = chunk_markdown(text, "demo.md")
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "title: Demo" in joined
    assert "body" in joined


def test_heading_level_jump_narrows_trail() -> None:
    text = "# A\n\na\n\n### C\n\nc\n\n## B\n\nb\n"
    headings = [chunk.heading for chunk in chunk_markdown(text, "jump.md")]
    assert "A" in headings
    assert "A > C" in headings
    assert "A > B" in headings
    assert "A > C > B" not in headings


def test_long_unbroken_word_is_hard_split() -> None:
    word = "x" * 500
    chunks = chunk_markdown(f"# T\n\n{word}\n", "blob.md", max_chars=100, overlap=10)
    assert len(chunks) >= 5
    assert all(len(chunk.text) <= 100 for chunk in chunks)
