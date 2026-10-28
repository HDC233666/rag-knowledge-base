"""向量索引：numpy 精确检索（默认兜底）或 FAISS HNSW 近似检索。

为什么要两套实现？
- 知识库只有几十到几百块时，一次矩阵乘法（numpy）就够，零额外依赖、结果精确；
- 规模上去后（上万块），精确检索的 O(N·D) 会变成瓶颈，换 FAISS 的 HNSW
  近似最近邻索引，查询复杂度接近对数级；代价是索引构建慢一点、结果是近似的。
- 两者接口一致（`search(query, top_k) -> [{...chunk, score}]`），评测/降级都不改上层。

用 `RAG_VECTOR_BACKEND` 切换：`auto`（有 faiss 就用，否则 numpy）/ `faiss` / `numpy`。

索引按"块内容指纹"缓存：文档不变时不用重复算 embedding、也不重建索引。
"""

from __future__ import annotations

import hashlib

import numpy as np

from .config import FAISS_EF_SEARCH, FAISS_HNSW_M, VECTOR_BACKEND
from .embeddings import get_embedder


def _fingerprint(chunks: list[dict]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for chunk in chunks:
        digest.update(str(chunk.get("id")).encode())
        digest.update((chunk.get("text") or "").encode())
    return digest.hexdigest()


class NumpyVectorIndex:
    """精确检索：把块向量拼成 (N, D) 矩阵，查询时矩阵乘查询向量。"""

    backend = "numpy"

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


class FaissVectorIndex:
    """近似检索：FAISS HNSW（内积 = 余弦，向量已归一化）。

    HNSW 参数：M 控制每个节点的连边数（越大越准、越占内存），
    ef_search 控制查询时搜索的候选数（越大越准、越慢）。
    对几十~几百块的小库来说 FAISS 不一定更快，真正的收益在万级规模。
    """

    backend = "faiss"

    def __init__(self, chunks: list[dict], embedder=None):
        import faiss  # 延迟导入：没装 faiss 时 numpy 后端不受影响

        self.embedder = embedder or get_embedder()
        self.chunks = list(chunks)
        self._faiss = faiss
        dim = self.embedder.dim if hasattr(self.embedder, "dim") else None
        if not self.chunks:
            self.index = None
            return
        vectors = self.embedder.embed([c.get("text", "") for c in self.chunks]).astype("float32")
        if dim is None:
            dim = vectors.shape[1]
        self.index = faiss.IndexHNSWFlat(int(dim), int(FAISS_HNSW_M), faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = int(FAISS_EF_SEARCH)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self.chunks or self.index is None:
            return []
        embed_query = getattr(self.embedder, "embed_query", None)
        query_vec = embed_query(query) if callable(embed_query) else self.embedder.embed([query])[0]
        query_vec = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        k = min(max(1, top_k), len(self.chunks))
        scores, indices = self.index.search(query_vec, k)
        hits = []
        for score, index in zip(scores[0], indices[0]):
            if int(index) < 0:  # HNSW 候选不足时返回 -1
                continue
            hits.append({**self.chunks[int(index)], "score": round(float(score), 4)})
        return hits


_BACKENDS = {"numpy": NumpyVectorIndex, "faiss": FaissVectorIndex}


def make_index(chunks: list[dict], embedder=None, backend: str | None = None):
    """按配置创建索引；faiss 不可用时自动退回 numpy（保证服务不挂）。"""
    choice = (backend or VECTOR_BACKEND or "auto").strip().lower()
    if choice == "numpy":
        return NumpyVectorIndex(chunks, embedder)
    if choice == "faiss":
        try:
            return FaissVectorIndex(chunks, embedder)
        except Exception as exc:  # faiss 未安装 / 版本问题 → 降级
            print(f"[vectorstore] faiss 索引创建失败，降级 numpy：{exc}", flush=True)
            return NumpyVectorIndex(chunks, embedder)
    # auto：优先 faiss，失败退回 numpy
    try:
        return FaissVectorIndex(chunks, embedder)
    except Exception:
        return NumpyVectorIndex(chunks, embedder)


_cache: dict[str, object] = {}


def vector_search(chunks: list[dict], query: str, top_k: int = 4, embedder=None) -> list[dict]:
    """向量检索入口：命中缓存则直接复用索引。"""
    if not chunks:
        return []
    key = f"{_fingerprint(chunks)}:{VECTOR_BACKEND}"
    index = _cache.get(key)
    if index is None:
        index = make_index(chunks, embedder)
        _cache.clear()  # 只保留当前知识库的索引，避免内存里堆旧版本
        _cache[key] = index
    return index.search(query, top_k)


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def backend_name() -> str:
    """当前实际使用的后端名（测试与 /api/health 展示用）。"""
    if _cache:
        return getattr(next(iter(_cache.values())), "backend", "numpy")
    if VECTOR_BACKEND == "numpy":
        return "numpy"
    if VECTOR_BACKEND == "faiss":
        return "faiss" if _faiss_available() else "numpy"
    # auto：装了 faiss 就用，否则 numpy（与 make_index 的降级逻辑保持一致）
    return "faiss" if _faiss_available() else "numpy"


def reset_cache() -> None:
    _cache.clear()
