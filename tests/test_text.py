import pytest

from obsidian_rag.text import ngrams, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Hello, World! RAG-2024") == ["hello", "world", "rag", "2024"]


def test_tokenize_empty() -> None:
    assert tokenize("") == []


def test_ngrams_unigrams() -> None:
    assert list(ngrams(["a", "b", "c"], 1)) == ["a", "b", "c"]


def test_ngrams_bigrams() -> None:
    assert list(ngrams(["a", "b", "c"], 2)) == ["a b", "b c"]


def test_ngrams_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        list(ngrams(["a"], 0))


def test_ngrams_longer_than_input_is_empty() -> None:
    assert list(ngrams(["a", "b"], 3)) == []
    assert list(ngrams([], 2)) == []
