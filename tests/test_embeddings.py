import numpy as np

from app.embeddings import HashEmbedder, get_embedder


def test_hash_embedder_is_normalized_and_deterministic():
    embedder = HashEmbedder()
    vectors = embedder.embed(["什么是 RAG", "检索增强生成"])
    assert vectors.shape == (2, 512)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    again = embedder.embed(["什么是 RAG"])
    assert np.allclose(vectors[0], again[0])


def test_get_embedder_respects_hash_backend(monkeypatch):
    monkeypatch.setenv("RAG_EMBED_BACKEND", "hash")
    assert isinstance(get_embedder(), HashEmbedder)
