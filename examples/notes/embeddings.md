# Embeddings

Every chunk is turned into a numeric vector, then stored so it can be compared
against a query vector using cosine similarity.

## Offline defaults

Two embedders work without a network connection or API key. The hashing
embedder maps word n-grams into a fixed-size signed vector. The TF-IDF embedder
fits a vocabulary over the corpus and weights tokens by inverse document
frequency.

## Optional dense vectors

`pip install 'obsidian-rag[embeddings]'` enables a sentence-transformers
backend for higher-quality dense vectors. The backend downloads a model on
first use, so it is opt-in.

## Similarity search

Vectors are L2-normalised at index time, which turns cosine similarity into a
plain dot product and makes retrieval a single matrix multiplication.
