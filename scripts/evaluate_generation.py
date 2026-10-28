"""生成质量评估：用「大模型当裁判」（LLM-as-a-judge）打分。

检索质量用 hit@k / MRR 衡量，但用户最终看到的是「答案」。
本脚本对一批问题实际走完整 RAG 流程，让裁判模型按三个维度打分：

    忠实度 faithfulness：答案里的结论是否都能在引用片段里找到依据（1-5）
    相关性 relevance：答案是否正面回答了问题（1-5）
    引用质量 citation：引用的片段是否真的支撑答案（1-5）

打分统一换算成 0-100 分的平均分，并统计「拒答数」（资料里没有就说不知道）。

用法（项目根目录，激活了 venv 的 python）：
    python scripts/evaluate_generation.py                 # 默认 20 题
    python scripts/evaluate_generation.py --limit 10      # 只跑前 10 题
    python scripts/evaluate_generation.py --judge deepseek-chat

说明：
    - 需要 DEEPSEEK_API_KEY，本脚本会真实调用接口（每题 2 次调用）。
    - 结果写入 data/eval/report_generation.json，同时打印表格。
    - 评分类 prompt 与指标定义参考 RAGAS（faithfulness / answer relevancy），
      但用单次 JSON 打分实现，零额外依赖，便于在面试里讲清原理。
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
from app.llm import LLMError, chat  # noqa: E402
from app.rag import _build_messages, retrieve  # noqa: E402

JUDGE_PROMPT = """你是 RAG 系统的严格评审。请只根据【引用片段】判断【回答】的质量。

评分维度（1-5 分，5 最好）：
1. faithfulness（忠实度）：回答中的关键结论是否都能在引用片段中找到依据？若有编造/外推，扣分。
2. relevance（相关性）：回答是否正面、完整地回答了问题？答非所问或空泛要扣分。
3. citation（引用质量）：引用的片段是否真的支撑回答，而不是凑数？

只输出一个 JSON 对象，不要输出其它文字：
{"faithfulness": 1-5, "relevance": 1-5, "citation": 1-5, "reason": "一句话理由"}"""


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted((ROOT / "data" / "sample").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for index, piece in enumerate(split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(
                {"id": f"{path.name}#{index}", "doc_name": path.name, "index": index, "text": piece}
            )
    return chunks


def parse_scores(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"裁判输出不是 JSON：{text[:120]}")
    data = json.loads(match.group(0))

    def clamp(key: str) -> float:
        value = float(data.get(key, 3))
        return max(1.0, min(5.0, value))

    return {
        "faithfulness": clamp("faithfulness"),
        "relevance": clamp("relevance"),
        "citation": clamp("citation"),
        "reason": str(data.get("reason", ""))[:80],
    }


def judge(question: str, answer_text: str, hits: list[dict]) -> dict:
    """裁判只看「检索回来的完整原文」，避免用 UI 展示用的 160 字摘要误判。"""
    evidence = "\n\n".join(
        f"[{index}] {hit['doc_name']} 原文：{hit['text'][:800]}"
        for index, hit in enumerate(hits, start=1)
    )
    payload = (
        f"【问题】\n{question}\n\n"
        f"【回答】\n{answer_text}\n\n"
        f"【引用片段】\n{evidence or '（无引用）'}"
    )
    result = chat(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": payload},
        ],
        temperature=0.1,
    )
    return parse_scores(result["text"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="评估题目数量（默认 20）")
    parser.add_argument("--no-generate", action="store_true", help="不真实生成，只复用 report 里的答案")
    args = parser.parse_args()

    chunks = load_chunks()
    spec = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = spec["questions"][: args.limit]

    rows: list[dict] = []
    refusals = 0
    started = time.perf_counter()

    for item in questions:
        question = item["question"]
        try:
            hits, trace = retrieve(question, chunks, 4)
        except (LLMError, ValueError) as exc:
            rows.append(
                {
                    "id": item["id"],
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if not hits:
            refusals += 1
            rows.append(
                {
                    "id": item["id"],
                    "question": question,
                    "answer": "知识库里没有找到相关内容。",
                    "citations": 0,
                    "latency_ms": 0,
                    "scores": {"faithfulness": 5.0, "relevance": 5.0, "citation": 5.0, "reason": "检索为空，正确拒答"},
                }
            )
            continue

        try:
            result = chat(_build_messages(question, hits))
        except (LLMError, ValueError) as exc:
            rows.append(
                {
                    "id": item["id"],
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        answer_text = result["text"]
        if answer_text.strip().startswith("知识库里没有找到") or "资料中没有" in answer_text[:40]:
            refusals += 1
        try:
            scores = judge(question, answer_text, hits)
        except (ValueError, LLMError) as exc:
            scores = {"faithfulness": None, "relevance": None, "citation": None, "reason": str(exc)[:80]}
        rows.append(
            {
                "id": item["id"],
                "question": question,
                "answer": answer_text,
                "citations": len(hits),
                "latency_ms": result.get("latency_ms"),
                "scores": scores,
            }
        )
        print(f"  ✓ {item['id']}  {json.dumps(scores, ensure_ascii=False)[:100]}")

    scored = [row for row in rows if row.get("scores") and row["scores"].get("faithfulness") is not None]
    total = len(scored)

    def mean(key: str) -> float:
        return sum(row["scores"][key] for row in scored) / total / 5 * 100 if total else 0.0

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": len(chunks),
        "questions": len(questions),
        "scored": total,
        "refusals": refusals,
        "scores_100": {
            "faithfulness": round(mean("faithfulness"), 1),
            "relevance": round(mean("relevance"), 1),
            "citation": round(mean("citation"), 1),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }

    out = ROOT / "data" / "eval" / "report_generation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n配置                忠实度   相关性   引用质量   拒答")
    print("-" * 56)
    print(
        f"{'generation':<20}{mean('faithfulness'):>7.1f}{mean('relevance'):>9.1f}"
        f"{mean('citation'):>10.1f}{refusals:>7d}/{len(questions)}"
    )
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
