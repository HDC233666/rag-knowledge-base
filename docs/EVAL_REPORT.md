# 评估报告（自动生成，请勿手改）

- 生成时间：2026-10-29 17:28:32
- 生成命令：`python scripts/eval_all.py`
- 说明：检索评估默认不调用大模型（除 `hybrid+rerank` 等配置需要 API）；生成质量与 RAGAS 需要 `--generation` / `--ragas`。

## 检索质量（基础语料）

- 语料切片：**14** 块（额外干扰语料：无）
- 题目数：**100**；embedding：`BAAI/bge-small-zh-v1.5`

| 配置 | hit@1 | hit@3 | MRR | 平均延迟 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.72 | 0.93 | 0.818 | 3 ms |
| hybrid | 0.67 | 0.91 | 0.790 | 12 ms |
| hybrid+rerank | 0.75 | 0.95 | 0.843 | 1492 ms |
| hybrid+rewrite+rerank | 0.75 | 0.95 | 0.844 | 2685 ms |

## 检索质量（千级干扰语料）

- 语料切片：**2114** 块（额外干扰语料：data/corpus/fastapi）
- 题目数：**100**；embedding：`BAAI/bge-small-zh-v1.5`

| 配置 | hit@1 | hit@3 | MRR | 平均延迟 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.61 | 0.87 | 0.733 | 188 ms |
| hybrid | 0.57 | 0.83 | 0.688 | 989 ms |
| hybrid+rerank | 0.76 | 0.93 | 0.842 | 1669 ms |
| hybrid+rewrite+rerank | 0.75 | 0.95 | 0.847 | 2061 ms |

