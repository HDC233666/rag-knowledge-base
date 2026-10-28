"""RAG 主流程 v2：

    问题 → 查询改写（多查询）→ 混合召回（BM25 + 向量 + RRF）
         → LLM 重排 → 拼上下文 → 大模型回答 → 答案 + 引用

每一步都能单独降级：
- 改写失败 / 关闭 → 只用原问题；
- 向量检索失败 → 只走 BM25；
- 重排失败 / 关闭 → 用 RRF 融合顺序。
所以这套链路在任何依赖缺失时都还能回答，只是效果会弱一些。
"""

from __future__ import annotations

from . import llm
from .config import CANDIDATE_K, TOP_K
from .rerank import rerank
from .retrieval import hybrid_search, rrf_fuse
from .rewrite import rewrite

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手，只根据【参考资料】回答。
规则：
1. 资料里没有的内容，直接回答「知识库里没有找到相关内容」，绝不编造。
2. 引用资料时在句末标注编号，例如 [1]。
3. 用中文回答，先给结论，再分点解释。"""


def _empty_answer() -> dict:
    return {
        "answer": "知识库里没有找到相关内容。请先上传文档，换个问法再试试。",
        "citations": [],
        "latency_ms": 0,
        "tokens": None,
        "trace": {"queries": [], "rewrite": False, "rerank": False, "candidates": 0},
    }


def retrieve(question: str, chunks: list[dict], top_k: int) -> tuple[list[dict], dict]:
    """检索 + 重排，返回 (命中片段, trace)。"""
    queries, used_rewrite = rewrite(question)

    # 每条查询各做一次混合召回，再把这些排名用 RRF 融合（多查询召回）
    ranked_lists = []
    for query in queries:
        hits = hybrid_search(chunks, query, top_k=CANDIDATE_K)
        if hits:
            ranked_lists.append(hits)
    candidates = rrf_fuse(ranked_lists)[:CANDIDATE_K] if ranked_lists else []

    hits, used_rerank = rerank(question, candidates, top_k)
    trace = {
        "queries": queries,
        "rewrite": used_rewrite,
        "rerank": used_rerank,
        "candidates": len(candidates),
    }
    return hits, trace


def answer(question: str, chunks: list[dict], top_k: int | None = None) -> dict:
    """回答一个问题。

    参数：
        question: 用户问题
        chunks: 所有已切分的知识块（DocumentStore.all_chunks()）
        top_k: 最终交给模型参考的片段数，None 用配置默认值
    返回：
        {answer, citations, latency_ms, tokens, trace}
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    hits, trace = retrieve(question, chunks, top_k or TOP_K)
    if not hits:
        return _empty_answer()

    context = "\n\n".join(
        f"[{i + 1}] 来源《{hit['doc_name']}》第 {hit['index'] + 1} 段：{hit['text']}"
        for i, hit in enumerate(hits)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】\n{question}"},
    ]
    result = llm.chat(messages)

    citations = [
        {
            "id": hit["id"],
            "doc_name": hit["doc_name"],
            "index": hit["index"],
            "score": hit["score"],
            "snippet": hit["text"][:160],
        }
        for hit in hits
    ]
    return {
        "answer": result["text"],
        "citations": citations,
        "latency_ms": result["latency_ms"],
        "tokens": result.get("tokens"),
        "trace": trace,
    }
