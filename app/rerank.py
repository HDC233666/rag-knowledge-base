"""重排 v3：支持两种打分后端，结果都进 SQLite 缓存。

为什么要重排：
- 召回阶段（BM25、向量、RRF）追求「别漏」，候选里混着大量弱相关内容；
- 重排阶段一次只看 10 来个候选，把真正对题的顶上来。

两种后端：
- llm（默认）：让大模型给候选打 0~10 分。零额外部署，但每次要一次 LLM 调用（约 1.3s）。
- cross：bge-reranker 交叉编码器（fastembed / ONNX 本地推理），一次几十毫秒、
  按对打分更准；失败时自动回退 llm。

两者结果都按「问题 + 候选指纹」缓存（RAG_ENABLE_CACHE=0 可关），
重复问题不会再付第二次延迟。
"""

from __future__ import annotations

import json
import re

from . import cache, llm
from .config import CROSS_ENCODER_MODEL, ENABLE_RERANK, RERANK_BACKEND

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


def _llm_scores(question: str, candidates: list[dict]) -> dict[int, float]:
    lines = []
    for i, hit in enumerate(candidates, start=1):
        snippet = (hit.get("text") or "")[:300].replace("\n", " ")
        lines.append(f"[{i}] 《{hit.get('doc_name', '')}》 {snippet}")
    payload = f"【问题】{question}\n\n【候选片段】\n" + "\n".join(lines)
    result = llm.chat(
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": payload}],
        temperature=0,
    )
    return _parse(result.get("text", ""))


_CROSS_ENCODER = None


def _cross_scores(question: str, candidates: list[dict]) -> dict[int, float]:
    """交叉编码器打分：模型直接看（问题, 片段）对，输出相关性分数。"""
    global _CROSS_ENCODER
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = TextCrossEncoder(model_name=CROSS_ENCODER_MODEL)
    texts = [(hit.get("text") or "")[:500] for hit in candidates]
    raw = list(_CROSS_ENCODER.rerank(question, texts))
    return {i: float(score) for i, score in enumerate(raw, start=1)}


def _fingerprint(candidates: list[dict]) -> str:
    """候选指纹：用 id + 分数标识，保证缓存只在候选集合一致时命中。"""
    return "|".join(f"{hit.get('id')}:{hit.get('score')}" for hit in candidates)


def rerank(
    question: str,
    candidates: list[dict],
    top_k: int,
    enabled: bool | None = None,
) -> tuple[list[dict], bool]:
    """返回 (重排后的 top_k 片段, 是否真的做了重排)。异常时保持原顺序。"""
    enabled = ENABLE_RERANK if enabled is None else enabled
    if not enabled or len(candidates) <= top_k:
        return candidates[:top_k], False

    backend = "cross" if RERANK_BACKEND == "cross" else "llm"
    cache_key = f"{question}\n{_fingerprint(candidates)}"

    scores: dict[int, float] = {}
    cached = cache.get(f"rerank:{backend}", cache_key)
    if cached and isinstance(cached.get("scores"), dict):
        try:
            scores = {int(k): float(v) for k, v in cached["scores"].items()}
        except (TypeError, ValueError):
            scores = {}

    if not scores:
        if backend == "cross":
            try:
                scores = _cross_scores(question, candidates)
            except Exception:  # noqa: BLE001
                scores = {}
        if not scores:
            try:
                scores = _llm_scores(question, candidates)
            except Exception:  # noqa: BLE001
                scores = {}
        if scores:
            cache.put(
                f"rerank:{backend}",
                cache_key,
                {"scores": {str(i): score for i, score in scores.items()}},
            )

    if not scores:
        return candidates[:top_k], False

    decorated = [
        {**hit, "rerank_score": scores.get(i, 0.0)}
        for i, hit in enumerate(candidates, start=1)
    ]
    decorated.sort(key=lambda item: item["rerank_score"], reverse=True)
    return decorated[:top_k], True
