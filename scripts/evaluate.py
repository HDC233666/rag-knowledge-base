"""离线评估：对比「纯 BM25 / 混合召回 / 混合+重排」的检索质量。

用法（项目根目录下，激活了 venv 的 python）：
    python scripts/evaluate.py                 # 20 题全跑
    python scripts/evaluate.py --limit 10      # 只跑前 10 题
    python scripts/evaluate.py --no-llm        # 跳过需要调用大模型的重排配置
    python scripts/evaluate.py --extra-corpus data/corpus/fastapi --report-suffix _large
        # 额外语料作为干扰文档一起参与检索，模拟「千级切片」的真实规模

指标：
    hit@1 / hit@3：正确片段是否出现在第 1 名 / 前 3 名
    MRR：正确片段排名的倒数的平均值（越接近 1 越好）

命中判定：检索片段的文档名与标注一致，且片段文本包含任一标注关键词。
结果同时写入 data/eval/report.json，方便写进 README。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台默认 GBK，避免 ✓ 之类字符报错
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import split_text  # noqa: E402
from app.config import CANDIDATE_K, CHUNK_OVERLAP, CHUNK_SIZE  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.rerank import rerank  # noqa: E402
from app.rag import retrieve as retrieve_pipeline  # noqa: E402
from app.retrieval import hybrid_search, search  # noqa: E402


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted((ROOT / "data" / "sample").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for index, piece in enumerate(split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(
                {
                    "id": f"{path.name}#{index}",
                    "doc_name": path.name,
                    "index": index,
                    "text": piece,
                }
            )
    return chunks


def load_extra_corpus(directory: str) -> list[dict]:
    """把额外语料目录下的 .txt 切块后作为干扰文档加入检索池。"""
    base = ROOT / directory
    chunks: list[dict] = []
    for path in sorted(base.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for index, piece in enumerate(split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(
                {
                    "id": f"corpus:{path.name}#{index}",
                    "doc_name": f"corpus:{path.name}",
                    "index": index,
                    "text": piece,
                }
            )
    return chunks


def is_hit(chunk: dict, item: dict) -> bool:
    return chunk.get("doc_name") == item["doc"] and any(
        keyword.lower() in (chunk.get("text") or "").lower()
        for keyword in item["keywords"]
    )


def reciprocal_rank(hits: list[dict], item: dict) -> float:
    for rank, chunk in enumerate(hits, start=1):
        if is_hit(chunk, item):
            return 1.0 / rank
    return 0.0


def run_config(name: str, retrieve, questions: list[dict], top_k: int = 4) -> dict:
    hit1 = hit3 = 0
    mrr = 0.0
    latencies: list[float] = []
    misses: list[str] = []

    for item in questions:
        started = time.perf_counter()
        hits = retrieve(item["question"])[:top_k]
        latencies.append((time.perf_counter() - started) * 1000)

        rank = reciprocal_rank(hits, item)
        mrr += rank
        if rank == 1.0:
            hit1 += 1
        if rank >= 1.0 / 3:
            hit3 += 1
        if rank == 0.0:
            misses.append(item["id"])

    total = len(questions)
    return {
        "config": name,
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "mrr": mrr / total,
        "avg_latency_ms": sum(latencies) / total,
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 题（0 = 全部）")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 重排配置")
    parser.add_argument("--extra-corpus", default="", help="额外语料目录（.txt），作为干扰文档")
    parser.add_argument("--report-suffix", default="", help="报告文件名后缀，如 _large")
    args = parser.parse_args()

    chunks = load_chunks()
    if args.extra_corpus:
        extra = load_extra_corpus(args.extra_corpus)
        chunks.extend(extra)
        print(f"额外语料: {len(extra)} 个切片（{args.extra_corpus}）")
    spec = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.limit:
        questions = questions[: args.limit]

    print(f"知识块: {len(chunks)} 个 | 题目: {len(questions)} 道 | 向量模型: {get_embedder().name}")

    configs = [
        ("bm25", lambda q: search(chunks, q, 4)),
        ("hybrid", lambda q: hybrid_search(chunks, q, 4)),
    ]
    if not args.no_llm:
        configs.append(
            (
                "hybrid+rerank",
                lambda q: rerank(q, hybrid_search(chunks, q, CANDIDATE_K), 4, enabled=True)[0],
            )
        )
        configs.append(
            (
                "hybrid+rewrite+rerank",
                lambda q: retrieve_pipeline(q, chunks, 4)[0],
            )
        )

    results = []
    for name, retrieve in configs:
        results.append(run_config(name, retrieve, questions))
        print(f"  ✓ {name} 完成")

    header = f"{'配置':<22}{'hit@1':>8}{'hit@3':>8}{'MRR':>8}{'平均延迟':>12}"
    print("\n" + header)
    print("-" * 60)
    for row in results:
        print(
            f"{row['config']:<22}{row['hit@1']:>8.2f}{row['hit@3']:>8.2f}"
            f"{row['mrr']:>8.3f}{row['avg_latency_ms']:>10.0f}ms"
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": len(chunks),
        "extra_corpus": args.extra_corpus or None,
        "questions": len(questions),
        "embedder": get_embedder().name,
        "results": results,
    }
    out = ROOT / "data" / "eval" / f"report{args.report_suffix}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
