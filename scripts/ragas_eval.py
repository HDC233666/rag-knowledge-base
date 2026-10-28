"""RAGAS 风格的生成质量评估：忠实度 / 答案相关性 / 上下文精度 / 上下文召回。

RAGAS（Retrieval Augmented Generation Assessment）是业界常用的 RAG 评估框架，
这里用「LLM 当裁判」的方式实现四个核心指标（不需要额外依赖）：

    忠实度 faithfulness       答案里的说法能否在检索片段中找到依据（0–1）
    答案相关性 answer_relevancy 答案是否切题、有没有跑题（0–10 归一化到 0–1）
    上下文精度 context_precision 检索到的片段里真正有用的比例（0–1）
    上下文召回 context_recall   标注的目标片段信息是否被检索结果覆盖（0–1）

用法：
    python scripts/ragas_eval.py --limit 20
    python scripts/ragas_eval.py --limit 20 --report-suffix _v1

结果写入 data/eval/ragas_report.json（或带后缀的文件）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import split_text  # noqa: E402
from app.config import CHUNK_OVERLAP, CHUNK_SIZE  # noqa: E402
from app.llm import chat  # noqa: E402
from app.rag import answer as rag_answer  # noqa: E402
from app.rag import retrieve as rag_retrieve  # noqa: E402

JUDGE_SYSTEM = (
    "你是严格的评测员，只输出 JSON，不要多余文字。"
    "判断时以给定的资料为准，不要使用你自己的知识。"
)


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


def find_reference(chunks: list[dict], item: dict) -> dict | None:
    for chunk in chunks:
        if chunk["doc_name"] == item["doc"] and any(
            keyword.lower() in chunk["text"].lower() for keyword in item["keywords"]
        ):
            return chunk
    return None


def judge(prompt: str) -> dict:
    result = chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    text = result["text"].strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"裁判没有输出 JSON: {text[:200]}")
    return json.loads(match.group(0))


def score_faithfulness(answer: str, context: str) -> float:
    data = judge(
        "任务：检查答案里的每个事实性说法能否在资料中找到依据。\n"
        f"资料：\n{context[:3000]}\n\n答案：\n{answer}\n\n"
        '输出 JSON：{"supported": 有依据的说法数, "total": 说法总数}。'
        "如果答案明确说资料里没有，记 total=1、supported=1。"
    )
    total = max(int(data.get("total", 0)), 1)
    supported = min(int(data.get("supported", 0)), total)
    return supported / total


def score_answer_relevancy(question: str, answer: str) -> float:
    data = judge(
        "任务：判断答案与问题的相关程度（0 = 完全跑题，10 = 完全切题）。\n"
        f"问题：{question}\n答案：{answer}\n\n"
        '输出 JSON：{"score": 0-10 的整数}。'
    )
    return max(0.0, min(10.0, float(data.get("score", 0)))) / 10.0


def score_context_precision(question: str, chunks: list[dict]) -> float:
    listing = "\n\n".join(
        f"[{i}] {chunk['text'][:600]}" for i, chunk in enumerate(chunks, start=1)
    )
    data = judge(
        "任务：判断每段资料对回答该问题是否有用（1 = 有用，0 = 没用）。\n"
        f"问题：{question}\n\n资料：\n{listing}\n\n"
        '输出 JSON：{"relevant": [按资料顺序的 0/1 数组]}。'
    )
    values = data.get("relevant") or []
    if not isinstance(values, list) or not values:
        return 0.0
    return sum(1 for v in values if int(v) == 1) / len(chunks)


def score_context_recall(question: str, reference: str, chunks: list[dict]) -> float:
    listing = "\n\n".join(chunk["text"][:600] for chunk in chunks)
    data = judge(
        "任务：判断「检索结果」是否覆盖了「参考答案」中回答该问题所需的信息（0-10 分）。\n"
        f"问题：{question}\n\n参考答案（标注的片段）：\n{reference[:1200]}\n\n"
        f"检索结果：\n{listing}\n\n"
        '输出 JSON：{"covered": 0-10 的整数}。'
    )
    return max(0.0, min(10.0, float(data.get("covered", 0)))) / 10.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="评估题目数（默认 20）")
    parser.add_argument("--report-suffix", default="", help="报告文件名后缀")
    args = parser.parse_args()

    chunks = load_chunks()
    spec = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = spec["questions"][: args.limit]

    rows = []
    started = time.perf_counter()
    for i, item in enumerate(questions, 1):
        question = item["question"]
        hits, trace = rag_retrieve(question, chunks, 4)
        result = rag_answer(question, chunks)
        answer_text = result["answer"]
        reference = find_reference(chunks, item)

        faith = score_faithfulness(answer_text, "\n\n".join(h["text"] for h in hits))
        relevancy = score_answer_relevancy(question, answer_text)
        precision = score_context_precision(question, hits)
        recall = score_context_recall(question, reference["text"], hits) if reference else 0.0

        rows.append(
            {
                "id": item["id"],
                "question": question,
                "faithfulness": round(faith, 3),
                "answer_relevancy": round(relevancy, 3),
                "context_precision": round(precision, 3),
                "context_recall": round(recall, 3),
                "answer_chars": len(answer_text),
                "retrieved": [chunk["doc_name"] for chunk in hits],
            }
        )
        print(
            f"  [{i:>2}/{len(questions)}] {item['id']:<4} "
            f"faith={faith:.2f} relevancy={relevancy:.2f} "
            f"precision={precision:.2f} recall={recall:.2f}"
        )

    def avg(key: str) -> float:
        return sum(row[key] for row in rows) / max(len(rows), 1)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": len(rows),
        "chunks": len(chunks),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "faithfulness": round(avg("faithfulness"), 3),
        "answer_relevancy": round(avg("answer_relevancy"), 3),
        "context_precision": round(avg("context_precision"), 3),
        "context_recall": round(avg("context_recall"), 3),
        "rows": rows,
    }

    out = ROOT / "data" / "eval" / f"ragas_report{args.report_suffix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nRAGAS 四指标（" + f"{len(rows)} 题）")
    print("-" * 44)
    for key, label in (
        ("faithfulness", "忠实度"),
        ("answer_relevancy", "答案相关性"),
        ("context_precision", "上下文精度"),
        ("context_recall", "上下文召回"),
    ):
        print(f"  {label:<8}{summary[key]:.3f}")
    print(f"\n耗时: {summary['elapsed_seconds']}s | 报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
