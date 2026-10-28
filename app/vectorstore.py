"""轻量向量索引：numpy 矩阵 + 余弦相似度。

为什么不用 FAISS / Chroma / Milvus？
- 知识库只有几十到几百个块时，一次矩阵乘法就够了（毫秒级）；
- 少一个依赖、少一个服务，部署和 CI 都简单；
- 规模上去后把 `VectorIndex.search` 换成 FAISS IndexFlatIP 或 Chroma 即可，
  接口保持不变（这也是面试时可以讲的取舍）。

索引按"块内容指纹"缓存：文档不变时不用重复算 embedding。
"""

from __future__ import annotations

import hashlib

import numpy as np

from .embeddings import get_embedder


def _fingerprint(chunks: list[dict]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for chunk in chunks:
        digest.update(str(chunk.get("id")).encode())
        digest.update((chunk.get("text") or "").encode())
    return digest.hexdigest()


class VectorIndex:
    """把块向量化成一个 (N, D) 矩阵；检索时矩阵乘查询向量。"""

    def __init__(self, chunks: list[dict], embedder=None):
        self.embedder = embedder or get_embedder()
        self.chunks = list(chunks)
        if self.chunks:
            self.matrix = self.embedder.embed([c.get("text", "") for c in self.chunks])
        else:
            self.matrix = np.zeros((0, 1), dtype=np.float32)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self.chunks:
            return []
        embed_query = getattr(self.embedder, "embed_query", None)
        query_vec = embed_query(query) if callable(embed_query) else self.embedder.embed([query])[0]
        scores = self.matrix @ query_vec  # 归一化向量点积 = 余弦相似度
        order = np.argsort(-scores)[: max(1, top_k)]
        return [
            {**self.chunks[int(i)], "score": round(float(scores[int(i)]), 4)}
            for i in order
        ]


_cache: dict[str, VectorIndex] = {}


def vector_search(chunks: list[dict], query: str, top_k: int = 4, embedder=None) -> list[dict]:
    """向量检索入口：命中缓存则直接复用索引。"""
    if not chunks:
        return []
    key = _fingerprint(chunks)
    index = _cache.get(key)
    if index is None:
        index = VectorIndex(chunks, embedder)
        _cache.clear()  # 只保留当前知识库的索引，避免内存里堆旧版本
        _cache[key] = index
    return index.search(query, top_k)
