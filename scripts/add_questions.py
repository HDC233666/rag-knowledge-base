"""把评估集从 62 题扩充到 100 题（q63-q100），并校验每条标注是否有效。

校验规则：
- id 唯一、doc 必须存在于 data/sample/*.md
- 每个 keyword 都必须出现在目标文档的全文里（大小写不敏感）
校验通过才写回 data/eval/questions.json，否则只打印错误、不落盘。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
EVAL = ROOT / "data" / "eval" / "questions.json"

NEW_QUESTIONS = [
    # vector-search.md
    ("q63", "vector-search.md", "embedding 向量在空间里有什么性质？", ["语义相近"]),
    ("q64", "vector-search.md", "为什么工程上要先把向量归一化？", ["归一化", "点积"]),
    ("q65", "vector-search.md", "bge-small-zh-v1.5 有什么特点？", ["512 维", "95MB"]),
    ("q66", "vector-search.md", "数据量到多大时该用 FAISS？", ["十万级"]),
    ("q67", "vector-search.md", "HNSW 是什么样的索引？", ["近似最近邻"]),
    ("q68", "vector-search.md", "百万级数据选什么向量库？", ["Milvus", "pgvector"]),
    ("q69", "vector-search.md", "对比学习里的负样本一般怎么来？", ["in-batch negative"]),
    ("q70", "vector-search.md", "InfoNCE 损失是怎么起作用的？", ["正样本", "负样本"]),
    ("q71", "vector-search.md", "换 embedding 模型时要注意什么？", ["全部重建"]),
    ("q72", "vector-search.md", "检索时为什么要给问题加指令前缀？", ["指令前缀"]),
    ("q73", "vector-search.md", "长文本直接编码有什么问题？", ["先切块"]),
    # hybrid-rerank.md
    ("q74", "hybrid-rerank.md", "重排有哪些常见做法？", ["交叉编码器", "LLM 打分"]),
    ("q75", "hybrid-rerank.md", "RRF 里的 k 一般取多少？", ["60"]),
    ("q76", "hybrid-rerank.md", "重排一般召回多少候选？", ["top-50"]),
    ("q77", "hybrid-rerank.md", "RRF 为什么不用把分数归一化？", ["名次"]),
    ("q78", "hybrid-rerank.md", "怎么避免调参只靠感觉？", ["hit@k", "MRR"]),
    # langgraph.md
    ("q79", "langgraph.md", "LangGraph 用哪些元素描述流程？", ["StateGraph"]),
    ("q80", "langgraph.md", "条件边是干什么的？", ["条件边"]),
    ("q81", "langgraph.md", "MemorySaver 和 SqliteSaver 有什么区别？", ["持久化"]),
    ("q82", "langgraph.md", "人工在环怎么实现？", ["human-in-the-loop"]),
    ("q83", "langgraph.md", "多智能体模式下子 Agent 的结果怎么汇总？", ["汇总"]),
    ("q84", "langgraph.md", "用 LangGraph 的代价是什么？", ["多一层抽象"]),
    # rag-evaluation.md
    ("q85", "rag-evaluation.md", "评估样本最少要标注什么？", ["期望命中的文档"]),
    ("q86", "rag-evaluation.md", "recall@k 适合什么场景？", ["多片段"]),
    ("q87", "rag-evaluation.md", "MRR 越接近 1 说明什么？", ["越靠前"]),
    ("q88", "rag-evaluation.md", "忠实度是怎么衡量的？", ["找到依据"]),
    ("q89", "rag-evaluation.md", "RAGAS 这类框架用在哪里？", ["工程化"]),
    ("q90", "rag-evaluation.md", "对比实验的结果应该记录哪些指标？", ["hit@1", "平均延迟"]),
    ("q91", "rag-evaluation.md", "hold-out 是干什么用的？", ["hold-out"]),
    # rag-quickstart.md
    ("q92", "rag-quickstart.md", "块太小会有什么问题？", ["一句话被切断"]),
    ("q93", "rag-quickstart.md", "块之间为什么要保留重叠？", ["跨块的句子"]),
    ("q94", "rag-quickstart.md", "BM25 需要训练吗？", ["不需要训练"]),
    ("q95", "rag-quickstart.md", "Chroma 和 Pinecone 分别适合什么场景？", ["轻量", "云服务"]),
    ("q96", "rag-quickstart.md", "提示词里怎么要求模型才不容易编造？", ["资料里没有就说不知道"]),
    # 干扰文档（验证不会乱答）
    ("q97", "cooking.md", "红烧肉一般炖多久？", ["60 分钟"]),
    ("q98", "cooking.md", "红烧肉怎么保存？", ["冷冻"]),
    ("q99", "travel-guide.md", "大熊猫基地建议几点入园？", ["8 点前"]),
    ("q100", "llm-interview.md", "function calling 输出的 JSON 里包含什么？", ["函数", "参数"]),
]


def main() -> int:
    spec = json.loads(EVAL.read_text(encoding="utf-8"))
    questions = spec["questions"]
    existing_ids = {item["id"] for item in questions}
    existing_questions = {item["question"] for item in questions}

    docs = {path.name: path.read_text(encoding="utf-8") for path in SAMPLE.glob("*.md")}
    errors: list[str] = []
    added = 0

    for qid, doc, question, keywords in NEW_QUESTIONS:
        if qid in existing_ids:
            errors.append(f"{qid}: id 重复")
            continue
        if question in existing_questions:
            errors.append(f"{qid}: 问题重复")
            continue
        if doc not in docs:
            errors.append(f"{qid}: doc 不存在 {doc}")
            continue
        missing = [kw for kw in keywords if kw.lower() not in docs[doc].lower()]
        if missing:
            errors.append(f"{qid}: 关键词未命中文档 {missing}")
            continue
        questions.append({"id": qid, "question": question, "doc": doc, "keywords": keywords})
        added += 1

    if errors:
        print("校验失败，未写入：")
        for err in errors:
            print("  ✗", err)
        return 1

    spec["questions"] = questions
    EVAL.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK：新增 {added} 题，评估集共 {len(questions)} 题")
    ids = [q["id"] for q in questions]
    print("id 范围:", ids[0], "→", ids[-1], "| 唯一:", len(ids) == len(set(ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
