import numpy as np
import pytest

from obsidian_rag.embedding import HashingEmbedder, TfidfEmbedder, build_embedder


def test_hashing_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=512)
    first = embedder.embed(["the cat sat"])
    second = embedder.embed(["the cat sat"])
    assert first.shape == (1, 512)
    np.testing.assert_allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-6)


def test_hashing_ranks_related_text_higher() -> None:
    embedder = HashingEmbedder(dim=2048)
    docs = embedder.embed(["the cat sat on the mat", "quantum chromodynamics particle physics"])
    query = embedder.embed(["cat mat"])[0]
    similarities = docs @ query
    assert similarities[0] > similarities[1]


def test_hashing_state_roundtrip() -> None:
    embedder = HashingEmbedder(dim=64, ngram_range=(1, 3))
    restored = HashingEmbedder.from_state(embedder.state())
    np.testing.assert_allclose(embedder.embed(["hello world"]), restored.embed(["hello world"]))


def test_hashing_empty_batch_has_expected_shape() -> None:
    vectors = HashingEmbedder(dim=32).embed([])
    assert vectors.shape == (0, 32)


def test_hashing_unigram_only_configuration() -> None:
    embedder = HashingEmbedder(dim=128, ngram_range=(1, 1))
    assert embedder.state()["ngram_range"] == [1, 1]


def test_hashing_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
    with pytest.raises(ValueError):
        HashingEmbedder(ngram_range=(0, 1))
    with pytest.raises(ValueError):
        HashingEmbedder(ngram_range=(3, 1))


def test_tfidf_fit_and_embed() -> None:
    embedder = TfidfEmbedder().fit(["cat sat mat", "dog ran fast", "cat and dog"])
    assert embedder.fitted
    assert embedder.dim > 0
    vector = embedder.embed(["cat mat"])
    assert vector.shape == (1, embedder.dim)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)


def test_tfidf_ranks_related_text_higher() -> None:
    corpus = ["the cat sat on the mat", "quantum chromodynamics particle physics"]
    embedder = TfidfEmbedder().fit(corpus)
    docs = embedder.embed(corpus)
    query = embedder.embed(["cat mat"])[0]
    similarities = docs @ query
    assert similarities[0] > similarities[1]


def test_tfidf_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed(["x"])
    with pytest.raises(RuntimeError):
        TfidfEmbedder().state()


def test_tfidf_state_roundtrip() -> None:
    embedder = TfidfEmbedder(max_features=10, min_df=1).fit(["cat sat mat", "dog ran fast"])
    restored = TfidfEmbedder.from_state(embedder.state())
    np.testing.assert_allclose(embedder.embed(["cat"]), restored.embed(["cat"]))


def test_tfidf_min_df_drops_rare_terms() -> None:
    embedder = TfidfEmbedder(min_df=2).fit(["cat dog", "cat bird"])
    assert "cat" in embedder.state()["vocab"]
    assert "bird" not in embedder.state()["vocab"]


def test_tfidf_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        TfidfEmbedder(max_features=0)
    with pytest.raises(ValueError):
        TfidfEmbedder(min_df=0)


def test_build_embedder_by_name() -> None:
    assert build_embedder("hashing", dim=32).name == "hashing"
    assert build_embedder("tfidf").name == "tfidf"
    with pytest.raises(ValueError):
        build_embedder("does-not-exist")


def test_sentence_transformer_requires_optional_extra() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            build_embedder("sentence-transformers")
    else:  # pragma: no cover - only when the optional extra is installed
        pytest.skip("sentence-transformers is installed")
