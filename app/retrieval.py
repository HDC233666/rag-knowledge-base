"""检索层：BM25（字面匹配）+ 向量（语义匹配）+ RRF 融合。

两路召回各有短板：
- BM25 精确、可解释，但同义词/换述说就查不到（"RAG" vs "检索增强生成"）；
- 向量检索懂语义，但专有名词、数字、缩写容易"懂过头"。
所以 v2 用混合召回：两路各取候选，再用 RRF（Reciprocal Rank Fusion）按排名融合，
最后交给 LLM 重排。RRF 只看名次、不看分数，天然解决了 BM25 分数和余弦分数不可比的问题。

检索流程：
    query → BM25 top-N ┐
          → 向量 top-N ┘→ RRF 融合 → top_k
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .config import CANDIDATE_K, RRF_K, TOP_K
from .vectorstore import vector_search

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


def rrf_fuse(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)，rank 从 1 开始。

    输入若干"已按相关度排好序"的命中列表（第一路 BM25、第二路向量…），
    输出按融合分降序的块列表。k 默认 60，是 RRF 论文里的经验值，
    作用是压低头部排名的绝对优势、让多路都靠前的结果更稳。
    """
    fused: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            cid = hit.get("id")
            item = fused.setdefault(
                cid,
                {**hit, "rrf": 0.0, "rrf_ranks": [], "sources": []},
            )
            item["rrf"] += 1.0 / (k + rank)
            item["rrf_ranks"].append(rank)
            item["sources"].append(hit.get("_source", "bm25"))
            # 保留两路原始分，展示用
            if "bm25_score" in hit:
                item.setdefault("bm25_score", hit["bm25_score"])
            if "vec_score" in hit:
                item["vec_score"] = hit["vec_score"]
    ordered = sorted(fused.values(), key=lambda item: item["rrf"], reverse=True)
    return [
        {**item, "rrf": round(item["rrf"], 6), "score": round(item["rrf"] * 100, 4)}
        for item in ordered
    ]


def hybrid_search(
    chunks: list[dict],
    query: str,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    embedder=None,
) -> list[dict]:
    """混合召回：BM25 与向量各取 candidate_k 个候选，RRF 融合后取 top_k。"""
    bm25_hits = [
        {**hit, "_source": "bm25", "bm25_score": hit["score"]}
        for hit in search(chunks, query, candidate_k)
    ]
    try:
        vector_hits = [
            {**hit, "_source": "vector", "vec_score": hit["score"]}
            for hit in vector_search(chunks, query, candidate_k, embedder)
        ]
    except Exception:  # embedding 失败不影响主流程：退化成纯 BM25
        vector_hits = []

    fused = rrf_fuse([bm25_hits, vector_hits])
    return fused[: max(1, top_k)]
