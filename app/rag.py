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

import time

from . import llm, metrics, safety
from .config import CANDIDATE_K, ENABLE_INJECTION_GUARD, TOP_K
from .rerank import rerank
from .retrieval import hybrid_search, rrf_fuse
from .rewrite import rewrite

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手，只根据【参考资料】回答。
规则：
1. 资料里没有的内容，直接回答「知识库里没有找到相关内容」，绝不编造。
2. 引用资料时在句末标注编号，例如 [1]。
3. 用中文回答，先给结论，再分点解释。
""" + safety.GUARD_RULES


def _empty_answer() -> dict:
    return {
        "answer": "知识库里没有找到相关内容。请先上传文档，换个问法再试试。",
        "citations": [],
        "latency_ms": 0,
        "tokens": None,
        "trace": {"queries": [], "rewrite": False, "rerank": False, "candidates": 0},
    }


def retrieve(
    question: str, chunks: list[dict], top_k: int, use_llm: bool = True
) -> tuple[list[dict], dict]:
    """检索 + 重排，返回 (命中片段, trace)。

    use_llm=False 时跳过「查询改写 / LLM 重排」，只跑混合召回 + RRF，
    用于评估基线、压测与调试（不消耗 API 费用）。
    """
    queries, used_rewrite = rewrite(question, enabled=use_llm and None)

    # 每条查询各做一次混合召回，再把这些排名用 RRF 融合（多查询召回）
    ranked_lists = []
    for query in queries:
        hits = hybrid_search(chunks, query, top_k=CANDIDATE_K)
        if hits:
            ranked_lists.append(hits)
    candidates = rrf_fuse(ranked_lists)[:CANDIDATE_K] if ranked_lists else []

    hits, used_rerank = rerank(question, candidates, top_k, enabled=use_llm and None)
    trace = {
        "queries": queries,
        "rewrite": used_rewrite,
        "rerank": used_rerank,
        "candidates": len(candidates),
    }
    return hits, trace


def _history_messages(history: list[dict] | None) -> list[dict]:
    """只保留最近 3 轮（user/assistant 交替），避免把上下文塞爆。"""
    messages = []
    for turn in (history or [])[-6:]:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:2000]})
    return messages


def _build_messages(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> tuple[list[dict], list[str]]:
    """拼 prompt：资料先中和注入片段，再用「不可信数据」边界包起来。

    返回 (messages, 命中的注入规则名)。规则名会写进 trace，方便事后追溯是哪份资料可疑。
    """
    flagged: list[str] = []
    lines: list[str] = []
    for i, hit in enumerate(hits):
        text = hit["text"]
        if ENABLE_INJECTION_GUARD:
            for name in safety.scan(text):
                if name not in flagged:
                    flagged.append(name)
            text = safety.neutralize(text)
        lines.append(f"[{i + 1}] 来源《{hit['doc_name']}》第 {hit['index'] + 1} 段：{text}")

    context = safety.wrap_context(lines) if ENABLE_INJECTION_GUARD else "\n\n".join(lines)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history_messages(history),
        {"role": "user", "content": f"{context}\n\n【问题】\n{question}"},
    ]
    return messages, flagged


def _citations(hits: list[dict]) -> list[dict]:
    return [
        {
            "id": hit["id"],
            "doc_name": hit["doc_name"],
            "index": hit["index"],
            "score": hit["score"],
            "snippet": hit["text"][:160],
        }
        for hit in hits
    ]


def answer(question: str, chunks: list[dict], top_k: int | None = None, history: list[dict] | None = None) -> dict:
    """回答一个问题。

    参数：
        question: 用户问题
        chunks: 所有已切分的知识块（DocumentStore.all_chunks()）
        top_k: 最终交给模型参考的片段数，None 用配置默认值
        history: 多轮对话历史 [{role, content}]，只取最近 3 轮
    返回：
        {answer, citations, latency_ms, tokens, trace}
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    hits, trace = retrieve(question, chunks, top_k or TOP_K)
    if not hits:
        return _empty_answer()

    messages, flagged = _build_messages(question, hits, history)
    if flagged:
        metrics.incr("rag_injection_suspect")
        trace["injection"] = flagged[:5]

    result = llm.chat(messages)
    answer_text = result["text"]
    usage = result.get("usage") or {}
    if usage:
        metrics.add_tokens(usage.get("prompt", 0), usage.get("completion", 0))
    if ENABLE_INJECTION_GUARD:
        answer_text, leaked = safety.scrub_output(answer_text, SYSTEM_PROMPT)
        if leaked:
            metrics.incr("rag_prompt_leak_blocked")
            trace["prompt_leak_blocked"] = True

    return {
        "answer": answer_text,
        "citations": _citations(hits),
        "latency_ms": result["latency_ms"],
        "tokens": result.get("tokens"),
        "usage": usage,
        "trace": trace,
    }


def answer_stream(question: str, chunks: list[dict], top_k: int | None = None, history: list[dict] | None = None):
    """流式回答：yield 事件字典，供接口层原样转成 SSE。

    事件顺序：
        {"type": "trace", ...}           检索完成（前端可提前显示进度）
        {"type": "delta", "text": "..."} 回答正文增量（可能很多个）
        {"type": "done", ...}            结束：完整回答 + 引用 + 耗时 + token
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    yield {"type": "stage", "text": "正在改写问题并检索知识库"}
    hits, trace = retrieve(question, chunks, top_k or TOP_K)
    if not hits:
        yield {"type": "trace", "trace": trace}
        payload = _empty_answer()
        yield {"type": "done", **{k: v for k, v in payload.items()}}
        return

    messages, flagged = _build_messages(question, hits, history)
    if flagged:
        metrics.incr("rag_injection_suspect")
        trace["injection"] = flagged[:5]

    yield {"type": "trace", "trace": trace}
    yield {"type": "stage", "text": f"已找到 {len(hits)} 个相关片段，正在生成回答"}
    started = time.time()
    parts: list[str] = []
    tokens = None
    usage: dict = {}
    for event in llm.chat_stream(messages):
        if "delta" in event:
            parts.append(event["delta"])
            yield {"type": "delta", "text": event["delta"]}
        elif event.get("done"):
            tokens = event.get("tokens")
            usage = event.get("usage") or {}

    answer_text = "".join(parts)
    if usage:
        metrics.add_tokens(usage.get("prompt", 0), usage.get("completion", 0))
    if ENABLE_INJECTION_GUARD:
        # 流式已经把原文发出去了，这里只能事后检查：命中泄露时在 done 事件里给出打码版本，
        # 前端以 done.answer 为准覆盖显示（文档里已注明这个限制）。
        answer_text, leaked = safety.scrub_output(answer_text, SYSTEM_PROMPT)
        if leaked:
            metrics.incr("rag_prompt_leak_blocked")
            trace["prompt_leak_blocked"] = True

    yield {
        "type": "done",
        "answer": answer_text,
        "citations": _citations(hits),
        "latency_ms": int((time.time() - started) * 1000),
        "tokens": tokens,
        "usage": usage,
        "trace": trace,
    }
