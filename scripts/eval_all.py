"""一条命令跑完全部离线评估，并生成一份可直接贴进 README/简历的报告。

为什么要这个脚本
    面试官不会逐条跑你的脚本，但会问「这些数字怎么来的」。把四类评估固定成
    一条命令、一个报告文件，等于把「可复现」变成项目的一部分：
        python scripts/eval_all.py              # 检索评估 + 汇总报告（不花钱）
        python scripts/eval_all.py --large      # 额外把千级语料当干扰文档
        python scripts/eval_all.py --generation # 追加生成质量（LLM-as-judge，会产生费用）
        python scripts/eval_all.py --ragas      # 追加 RAGAS 四指标（会产生费用）

产物
    data/eval/report.json            检索评估原始数据（evaluate.py 产出）
    data/eval/report_large.json      千级语料版本（--large）
    data/eval/generation_report.json 生成质量（--generation）
    data/eval/ragas_report.json      RAGAS（--ragas）
    docs/EVAL_REPORT.md              汇总报告（每次运行覆盖）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
REPORT_MD = ROOT / "docs" / "EVAL_REPORT.md"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"\n$ {' '.join(Path(c).name if i == 1 else c for i, c in enumerate(cmd))}")
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT)  # noqa: S603 - 只跑本仓库脚本
    elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(f"{script} 失败（退出码 {result.returncode}）")
    print(f"  ✓ {script} 用时 {elapsed:.1f}s")


def load(name: str) -> dict | None:
    path = EVAL_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _retrieval_table(report: dict) -> list[str]:
    lines = [
        f"- 语料切片：**{report.get('chunks')}** 块（额外干扰语料：{report.get('extra_corpus') or '无'}）",
        f"- 题目数：**{report.get('questions')}**；embedding：`{report.get('embedder')}`",
        "",
        "| 配置 | hit@1 | hit@3 | MRR | 平均延迟 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("results", []):
        lines.append(
            f"| {row['config']} | {row['hit@1']:.2f} | {row['hit@3']:.2f} "
            f"| {row['mrr']:.3f} | {row['avg_latency_ms']:.0f} ms |"
        )
    return lines


def _generation_section(report: dict) -> list[str]:
    lines = ["## 生成质量（LLM-as-a-judge）", ""]
    summary = report.get("summary") or {}
    if summary:
        lines += [
            "| 指标 | 分数 |",
            "| --- | ---: |",
            *[f"| {key} | {value} |" for key, value in summary.items()],
        ]
    else:
        lines.append(f"- 题目数：{report.get('questions', len(report.get('items', [])))}")
    lines.append("")
    lines.append("> 判定模型与生成模型分开调用，评分维度见 `scripts/evaluate_generation.py`。")
    return lines


def _ragas_section(report: dict) -> list[str]:
    keys = [
        ("faithfulness", "忠实度"),
        ("answer_relevancy", "答案相关性"),
        ("context_precision", "上下文精度"),
        ("context_recall", "上下文召回"),
    ]
    lines = ["## RAGAS 四指标", "", "| 指标 | 分数 |", "| --- | ---: |"]
    for key, label in keys:
        value = report.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"| {label}（{key}） | {value:.3f} |")
    lines += ["", f"- 题目数：{report.get('questions', 'n/a')}"]
    return lines


def build_markdown(sections: list[str]) -> str:
    head = [
        "# 评估报告（自动生成，请勿手改）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 生成命令：`python scripts/eval_all.py`",
        "- 说明：检索评估默认不调用大模型（除 `hybrid+rerank` 等配置需要 API）；生成质量与 RAGAS 需要 `--generation` / `--ragas`。",
        "",
    ]
    return "\n".join([*head, *sections, ""])


def main() -> int:
    parser = argparse.ArgumentParser(description="一条命令跑完全部评估并生成报告")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（快速验证）")
    parser.add_argument("--large", action="store_true", help="加入千级语料作为干扰文档")
    parser.add_argument("--no-llm", action="store_true", help="跳过需要调用大模型的检索配置")
    parser.add_argument("--generation", action="store_true", help="追加生成质量评估（花钱）")
    parser.add_argument("--ragas", action="store_true", help="追加 RAGAS 评估（花钱）")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []

    base_args: list[str] = []
    if args.limit:
        base_args += ["--limit", str(args.limit)]
    if args.no_llm:
        base_args += ["--no-llm"]

    run("evaluate.py", *base_args)
    base = load("report.json")
    if base:
        sections += ["## 检索质量（基础语料）", "", *_retrieval_table(base), ""]

    if args.large:
        run("evaluate.py", *base_args, "--extra-corpus", "data/corpus/fastapi", "--report-suffix", "_large")
        large = load("report_large.json")
        if large:
            sections += ["## 检索质量（千级干扰语料）", "", *_retrieval_table(large), ""]

    if args.generation:
        run("evaluate_generation.py")
        generation = load("generation_report.json")
        if generation:
            sections += [*_generation_section(generation), ""]

    if args.ragas:
        run("ragas_eval.py")
        ragas = load("ragas_report.json")
        if ragas:
            sections += [*_ragas_section(ragas), ""]

    REPORT_MD.write_text(build_markdown(sections), encoding="utf-8")
    print(f"\n汇总报告已写入: {REPORT_MD}")
    print("把这篇文章里的表格贴进 README，或直接截「一条命令 + 报告」的图放进简历。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
