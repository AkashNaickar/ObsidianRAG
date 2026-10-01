"""Small, dependency-free text helpers shared by the embedders."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    """Yield contiguous ``n``-grams joined by a single space."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        yield from tokens
        return
    for index in range(len(tokens) - n + 1):
        yield " ".join(tokens[index : index + n])
