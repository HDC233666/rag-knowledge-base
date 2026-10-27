"""RAG 主流程：检索知识库 → 拼上下文 → 让大模型回答 → 返回答案 + 引用。"""

from __future__ import annotations

from . import llm
from .config import TOP_K
from .retrieval import search

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手，只根据【参考资料】回答。
规则：
1. 资料里没有的内容，直接回答「知识库里没有找到相关内容」，绝不编造。
2. 引用资料时在句末标注编号，例如 [1]。
3. 用中文回答，先给结论，再分点解释。"""


def answer(question: str, chunks: list[dict], top_k: int | None = None) -> dict:
    """回答一个问题。

    参数：
        question: 用户问题
        chunks: 所有已切分的知识块（DocumentStore.all_chunks()）
        top_k: 检索返回的片段数，None 用配置默认值
    返回：
        {answer, citations, latency_ms, tokens}
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    hits = search(chunks, question, top_k or TOP_K)
    if not hits:
        return {
            "answer": "知识库里没有找到相关内容。请先上传文档，换个问法再试试。",
            "citations": [],
            "latency_ms": 0,
            "tokens": None,
        }

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
    }
