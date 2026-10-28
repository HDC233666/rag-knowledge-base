"""文本向量化：优先 bge-small-zh（fastembed + ONNX），装不上/下不动时自动退化为哈希向量。

为什么选 bge-small-zh-v1.5：
- 中文语义检索效果好，512 维，模型约 95MB，CPU 就能跑；
- fastembed 用 ONNX Runtime，不依赖 torch，安装和启动都轻；
- 想换模型（bge-m3 / OpenAI text-embedding-3）改 `RAG_EMBED_MODEL` 或这里一处即可。

降级策略：模型加载失败时用 HashEmbedder（字符 n-gram 哈希，零依赖），
保证测试和离线环境也能跑通整个链路，只是没有"语义"能力。
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Sequence

import numpy as np

DIM = 512
MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
# bge 中文模型官方建议：给「查询」加指令前缀，文档侧不加，检索效果更好
QUERY_PREFIX = os.getenv("RAG_EMBED_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章：")


def _normalize(rows: np.ndarray) -> np.ndarray:
    """L2 归一化：归一化后两个向量的点积就是余弦相似度。"""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (rows / norms).astype(np.float32)


class HashEmbedder:
    """零依赖兜底：字符 1~3-gram 哈希到 512 维。

    只有字面相似、没有语义能力，仅用于测试 / 离线环境。
    """

    name = "hash-ngram-512"
    dim = DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = (text or "").lower()
            for n in (1, 2, 3):
                for j in range(len(lowered) - n + 1):
                    gram = lowered[j : j + n].encode("utf-8")
                    h = int.from_bytes(hashlib.blake2b(gram, digest_size=8).digest(), "little")
                    rows[i, h % self.dim] += 1.0
        return _normalize(rows)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class FastEmbedEmbedder:
    """bge-small-zh 的 ONNX 推理封装（首次调用会加载/下载模型）。"""

    def __init__(self, model_name: str = MODEL_NAME):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.name = model_name
        self.dim = DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        return _normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """检索查询要加 bge 的指令前缀；文档块走 embed() 不加。"""
        return self.embed([QUERY_PREFIX + (query or "")])[0]


_lock = threading.Lock()
_embedder: FastEmbedEmbedder | HashEmbedder | None = None


def get_embedder():
    """进程内单例。设置 RAG_EMBED_BACKEND=hash 可强制使用降级实现（测试/CI 用）。"""
    global _embedder
    if os.getenv("RAG_EMBED_BACKEND", "auto").lower() == "hash":
        return HashEmbedder()
    if _embedder is None:
        with _lock:
            if _embedder is None:
                try:
                    _embedder = FastEmbedEmbedder()
                except Exception:  # 模型缺失 / 网络不通：降级，不影响主流程
                    _embedder = HashEmbedder()
    return _embedder
