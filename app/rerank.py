"""LLM 重排：让模型给候选片段按「能不能回答这个问题」打分，再取 top_k。

为什么要重排：
- 召回阶段（BM25、向量、RRF）追求"别漏"，候选里混着大量弱相关内容；
- 重排阶段一次只看 10 来个候选，能把真正对题的顶上来。

生产里常用 cross-encoder（如 bge-reranker-base）做重排；这里用 LLM 打分替代，
好处是不用再部署一个模型，缺点是多一次调用、延迟高一些，适合知识库规模不大的场景。
关掉：RAG_ENABLE_RERANK=0。
"""

from __future__ import annotations

import json
import re

from . import llm
from .config import ENABLE_RERANK

PROMPT = """你是检索结果重排器。根据【问题】，给每个【候选片段】打 0~10 分：
10 = 片段本身就能直接回答问题；5 = 相关但不完整；0 = 完全无关。
只输出 JSON，不要解释，格式：{"scores": [{"i": 1, "score": 9}, {"i": 2, "score": 3}]}"""


def _parse(raw: str) -> dict[int, float]:
    match = re.search(r"\{.*\}", (raw or "").strip(), re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    scores: dict[int, float] = {}
    for item in data.get("scores", []) if isinstance(data, dict) else []:
        try:
            scores[int(item["i"])] = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return scores


def rerank(
    question: str,
    candidates: list[dict],
    top_k: int,
    enabled: bool | None = None,
) -> tuple[list[dict], bool]:
    """返回 (重排后的 top_k 片段, 是否真的调用了模型)。异常时保持原顺序。"""
    enabled = ENABLE_RERANK if enabled is None else enabled
    if not enabled or len(candidates) <= top_k:
        return candidates[:top_k], False

    lines = []
    for i, hit in enumerate(candidates, start=1):
        snippet = (hit.get("text") or "")[:300].replace("\n", " ")
        lines.append(f"[{i}] 《{hit.get('doc_name', '')}》 {snippet}")
    payload = f"【问题】{question}\n\n【候选片段】\n" + "\n".join(lines)

    try:
        result = llm.chat(
            [{"role": "system", "content": PROMPT}, {"role": "user", "content": payload}],
            temperature=0,
        )
    except Exception:
        return candidates[:top_k], False

    scores = _parse(result.get("text", ""))
    if not scores:
        return candidates[:top_k], False

    decorated = [
        {**hit, "rerank_score": scores.get(i, 0.0)}
        for i, hit in enumerate(candidates, start=1)
    ]
    decorated.sort(key=lambda item: item["rerank_score"], reverse=True)
    return decorated[:top_k], True
