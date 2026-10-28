"""并发压测脚本：测检索接口或健康检查接口的延迟分布与吞吐。

用法：
    python scripts/bench.py --path /api/retrieve --total 100 --concurrency 20
    python scripts/bench.py --path /api/stats --total 200 --concurrency 50

只依赖 httpx（项目已有），不消耗大模型 API 费用（默认 llm=false）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

import httpx

QUESTIONS = [
    "BM25 和向量检索有什么区别？",
    "为什么要做混合检索？",
    "RRF 的公式是什么？",
    "交叉编码器为什么比双塔慢？",
    "RAG 的评估指标有哪些？",
    "怎么降低大模型幻觉？",
    "文档切分块大小怎么选？",
    "HNSW 索引适合什么场景？",
]


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


async def run(args: argparse.Namespace) -> int:
    latencies: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()

    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        # 预热：先把模型/缓存/连接热起来，避免首请求的加载时间污染统计
        for warm in range(args.warmup):
            if args.path == "/api/retrieve":
                await client.post(
                    "/api/retrieve",
                    json={
                        "question": QUESTIONS[warm % len(QUESTIONS)],
                        "collection": args.collection,
                        "top_k": 4,
                        "llm": False,
                    },
                )
            else:
                await client.get(args.path)

        async def one(i: int) -> None:
            nonlocal errors
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    if args.path == "/api/retrieve":
                        payload = {
                            "question": QUESTIONS[i % len(QUESTIONS)],
                            "collection": args.collection,
                            "top_k": 4,
                            "llm": args.llm,
                        }
                        resp = await client.post(args.path, json=payload)
                    else:
                        resp = await client.get(args.path)
                    if resp.status_code >= 400:
                        errors += 1
                except Exception:  # noqa: BLE001 - 压测只关心成功率
                    errors += 1
                finally:
                    latencies.append((time.perf_counter() - t0) * 1000)

        await asyncio.gather(*(one(i) for i in range(args.total)))

    elapsed = time.perf_counter() - started
    ok = len(latencies) - errors
    latencies.sort()

    summary = {
        "path": args.path,
        "total": args.total,
        "concurrency": args.concurrency,
        "ok": ok,
        "errors": errors,
        "elapsed_s": round(elapsed, 2),
        "qps": round(ok / elapsed, 1) if elapsed else 0,
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p90": round(percentile(latencies, 0.90), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "p99": round(percentile(latencies, 0.99), 1),
            "max": round(max(latencies), 1) if latencies else 0,
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 服务并发压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--path", default="/api/retrieve")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--llm", action="store_true", help="retrieve 时启用查询改写/重排（会花钱，慎用）")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
