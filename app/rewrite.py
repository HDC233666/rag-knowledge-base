"""查询改写：把用户口语化的问题扩写成 1~3 条检索查询（多查询召回）。

一个查询只能命中一种说法，用户问「RAG 是啥」时，文档里写的可能是
「检索增强生成」。让模型先补同义说法 / 拆子问题，再分别去检索，
能明显提升召回率——这就是「Multi-Query Retrieval」。

代价是多一次 LLM 调用（几百 token），可以在 .env 里用 RAG_ENABLE_REWRITE=0 关掉。
"""

from __future__ import annotations

import json
import re

from . import cache, llm
from .config import ENABLE_REWRITE

PROMPT = """你是检索查询优化助手。把用户问题改写成 1~3 条适合做检索的短查询。
要求：
1. 第一条保留原问题；
2. 其余几条补充同义说法、上位词或拆分后的子问题；
3. 每条不超过 40 字，不要加入问题里没有的事实。
只输出 JSON，不要解释、不要代码块，格式：{"queries": ["原问题", "同义改写"]}"""


def _parse(raw: str) -> list[str]:
    """从模型输出里尽量稳地抠出 queries 列表。"""
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            items = data.get("queries") if isinstance(data, dict) else None
            if isinstance(items, list):
                return [str(item) for item in items]
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


def rewrite(question: str, enabled: bool | None = None) -> tuple[list[str], bool]:
    """返回 (查询列表, 是否真的调用了模型)。

    任何异常（没 API Key、网络、输出格式不对）都静默降级为「只用原问题」，
    保证检索链路永远可用。
    """
    question = (question or "").strip()
    if not question:
        return [], False
    enabled = ENABLE_REWRITE if enabled is None else enabled
    if not enabled:
        return [question], False

    # 先查缓存：同一个问题改写结果不会变，命中就省一次 LLM 调用
    cached = cache.get("rewrite", question)
    if cached and isinstance(cached.get("queries"), list):
        queries = [
            str(item).strip().strip("「」\"'").strip()
            for item in cached["queries"]
            if str(item).strip()
        ]
        if queries:
            return queries[:3], True

    try:
        result = llm.chat(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
    except Exception:
        return [question], False

    queries: list[str] = []
    for item in _parse(result.get("text", "")):
        item = item.strip().strip("「」\"'").strip()
        if 2 <= len(item) <= 80 and item.lower() not in {q.lower() for q in queries}:
            queries.append(item)

    if not queries:
        return [question], False
    if question.lower() not in {q.lower() for q in queries}:
        queries.insert(0, question)
    queries = queries[:3]
    cache.put("rewrite", question, {"queries": queries})
    return queries, True
