"""极简指标收集：进程内计数器 + 延迟分位数 + token 用量，供 /api/metrics（Prometheus 文本）
和 /api/dashboard（JSON，给前端/面试演示看）使用。

为什么自己写
    引 prometheus_client 要多一个依赖，而这里只需要「计数 + 分位数 + 求和」三件事。
    多线程安全靠一把锁；延迟样本用固定长度的环形窗口（默认 2000 条），
    内存有上限，统计的是「最近一段时间的 P50/P95/P99」，和真实监控系统的滚动窗口一致。

成本估算
    DeepSeek 按 token 计费，输入（缓存未命中）2 元/M、输入（命中）0.2 元/M、输出 8 元/M。
    单价都能用环境变量覆盖（见 config.PRICE_*），所以换个网关也不用改代码。
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque

from . import config

_lock = threading.Lock()
_counters: Counter[str] = Counter()
_gauges: dict[str, float] = {}
_latencies: dict[str, deque[float]] = {}
_tokens = {"prompt": 0, "completion": 0}
_LATENCY_WINDOW = 2000
_created_at = time.time()


def incr(name: str, value: int = 1) -> None:
    with _lock:
        _counters[name] += value


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _gauges[name] = float(value)


def observe_latency(name: str, ms: float) -> None:
    """记录一次耗时（毫秒），只保留最近 _LATENCY_WINDOW 个样本。"""
    bucket = _latencies.setdefault(name, deque(maxlen=_LATENCY_WINDOW))
    with _lock:
        bucket.append(float(ms))


def add_tokens(prompt: int = 0, completion: int = 0) -> None:
    """累加 token 用量（来自接口返回的 usage）。"""
    with _lock:
        _tokens["prompt"] += max(0, int(prompt or 0))
        _tokens["completion"] += max(0, int(completion or 0))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def latency_stats(name: str) -> dict:
    """返回 {count, p50, p95, p99, max}（毫秒）。"""
    with _lock:
        values = sorted(_latencies.get(name, ()))
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "count": len(values),
        "p50": round(_quantile(values, 0.50), 1),
        "p95": round(_quantile(values, 0.95), 1),
        "p99": round(_quantile(values, 0.99), 1),
        "max": round(values[-1], 1),
        "avg": round(sum(values) / len(values), 1),
    }


def cost_estimate(stats: dict | None = None) -> dict:
    """按 token 估算花费（元）。cached 部分按「输入命中价」算，未提供时按全价。"""
    prompt = _tokens["prompt"]
    completion = _tokens["completion"]
    cached = 0
    if stats:
        cached = int(stats.get("counters", {}).get("rag_tokens_prompt_cached", 0))
    cached = min(cached, prompt)
    fresh = prompt - cached
    amount = (
        fresh / 1_000_000 * config.PRICE_INPUT_PER_M
        + cached / 1_000_000 * config.PRICE_INPUT_CACHED_PER_M
        + completion / 1_000_000 * config.PRICE_OUTPUT_PER_M
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_prompt_tokens": cached,
        "estimated_cny": round(amount, 4),
        "price_per_million": {
            "input": config.PRICE_INPUT_PER_M,
            "input_cached": config.PRICE_INPUT_CACHED_PER_M,
            "output": config.PRICE_OUTPUT_PER_M,
        },
    }


def snapshot() -> dict:
    with _lock:
        return {
            "counters": dict(_counters),
            "gauges": dict(_gauges),
            "tokens": dict(_tokens),
            "uptime_seconds": round(time.time() - _created_at, 1),
        }


def dashboard() -> dict:
    """给 /api/dashboard 的结构化视图：计数 + 延迟分位 + 缓存命中率 + 成本。"""
    data = snapshot()
    counters = data["counters"]
    cache_hit = counters.get("rag_cache_hit", 0)
    cache_miss = counters.get("rag_cache_miss", 0)
    cache_total = cache_hit + cache_miss
    return {
        "uptime_seconds": data["uptime_seconds"],
        "counters": counters,
        "gauges": data["gauges"],
        "latency_ms": {
            "ask": latency_stats("rag_ask"),
            "stream": latency_stats("rag_stream"),
            "retrieve": latency_stats("rag_retrieve"),
        },
        "cache": {
            "hit": cache_hit,
            "miss": cache_miss,
            "hit_rate": round(cache_hit / cache_total, 3) if cache_total else None,
        },
        "cost": cost_estimate(data),
    }


def reset() -> None:
    """测试用：清空所有指标。"""
    with _lock:
        _counters.clear()
        _gauges.clear()
        _latencies.clear()
        _tokens["prompt"] = 0
        _tokens["completion"] = 0


def render_prometheus() -> str:
    """输出 Prometheus 文本格式：counter / gauge / summary 三类混排。"""
    data = snapshot()
    lines: list[str] = []
    for name, value in sorted(data["counters"].items()):
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")
    for name, value in sorted(data["gauges"].items()):
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value:g}")

    # 延迟：summary 语义（count/sum + 分位数），足够 Grafana 直接画
    for name in sorted(_latencies):
        stats = latency_stats(name)
        if not stats["count"]:
            continue
        lines.append(f"# TYPE {name}_ms summary")
        lines.append(f"{name}_ms{{quantile=\"0.5\"}} {stats['p50']}")
        lines.append(f"{name}_ms{{quantile=\"0.95\"}} {stats['p95']}")
        lines.append(f"{name}_ms{{quantile=\"0.99\"}} {stats['p99']}")
        lines.append(f"{name}_ms_count {stats['count']}")

    cost = cost_estimate(data)
    lines.append("# TYPE rag_estimated_cost_cny gauge")
    lines.append(f"rag_estimated_cost_cny {cost['estimated_cny']}")
    lines.append("# TYPE rag_tokens_total counter")
    lines.append(f"rag_tokens_total {cost['prompt_tokens'] + cost['completion_tokens']}")

    if not lines:
        lines.append("# no metrics yet")
    return "\n".join(lines) + "\n"
