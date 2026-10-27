"""轻量检索：BM25（纯 Python 实现，不用下载模型、不用向量库）。

为什么用 BM25 而不是向量检索？
- 零依赖、几毫秒出结果，适合"上传文档立刻就能问答"的 demo；
- 中文按「单字 + 二元组」切词，避免没有分词器的问题；
- 生产环境想更准，可以换成 FAISS / Chroma / Milvus + embedding。

检索流程：把每个块切成 token → 建 BM25 索引 → 对问题同样切词 → 打分排序取 top_k。
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .config import TOP_K

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文按单字 + 相邻二元组切；英文/数字按单词切。"""
    text = text or ""
    tokens: list[str] = [m.group(0).lower() for m in _WORD.finditer(text)]
    cjk = "".join(_CJK.findall(text))
    tokens.extend(cjk)
    tokens.extend(cjk[i : i + 2] for i in range(len(cjk) - 1))
    return tokens


def search(
    chunks: list[dict],
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """在块列表里检索，返回带 score 的 top_k 块（按相关度降序）。"""
    if not chunks:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    corpus: list[list[str]] = []
    kept: list[dict] = []
    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        if tokens:
            corpus.append(tokens)
            kept.append(chunk)
    if not corpus:
        return []

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(kept, scores), key=lambda pair: pair[1], reverse=True)
    return [
        {**chunk, "score": round(float(score), 4)}
        for chunk, score in ranked[: max(1, top_k)]
    ]
