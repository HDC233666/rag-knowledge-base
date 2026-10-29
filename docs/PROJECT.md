# 项目一页纸：RAG 知识库问答系统

> 一句话：把「私有文档怎么变成能溯源的问答」做成一个可部署、可评估、可观测的服务，而不是一个调 API 的 demo。

## 1. 要解决的问题

大模型不知道公司内部文档、个人笔记、最新资料，直接问会一本正经地编。RAG 的做法是：先从资料里检索相关片段，再让模型基于片段回答，并给出引用出处。

这个项目关心的是**工程上把它做扎实**：检索质量怎么量化、答错了怎么查、上云怎么跑、被注入攻击怎么办。

## 2. 做了什么（按能力分层）

| 层 | 内容 |
| --- | --- |
| 文档处理 | 上传 txt / md / pdf / csv，按句切分（600 字 / 80 字重叠），多知识库隔离（collection） |
| 检索 | BM25 + bge-small-zh 向量双路召回，RRF 融合（k=60，候选 12），LLM 查询改写、LLM 重排（可切 cross-encoder） |
| 索引 | FAISS HNSW（M=32、efSearch=64），没装 faiss 自动降级 numpy 精确检索 |
| 服务 | FastAPI，同步 / 流式两套接口；SSE 推送 stage / trace / delta / done；每 IP 限流；每日演示额度 |
| 存储 | MySQL（documents / chunks，外键级联删除）→ 连不上自动降级 JSON 文件 |
| 缓存 | 改写与重排结果 SQLite TTL 缓存，减少重复 API 调用 |
| 安全 | 提示注入防护：硬模式中和 + 「不可信数据」边界 + 系统规则；回答含系统提示原文时打码 |
| 会话 | 服务端 session_id 记忆最近 6 轮（TTL 30 分钟），客户端不用再传 history |
| 可观测 | `/api/metrics`（Prometheus 文本）+ `/api/dashboard`（JSON：计数、P50/P95、缓存命中率、token 与成本估算） |
| 评估 | 100 题评估集 + `scripts/eval_all.py` 一条命令生成 `docs/EVAL_REPORT.md`；另有 RAGAS 与 LLM-as-judge |
| 交付 | Dockerfile、docker compose（可选 MySQL）、GitHub Actions（测试 + 镜像构建）、中文文档 4 篇 |

## 3. 结果（可复现）

检索质量（100 题，同一套数据）：

| 配置 | hit@1 | hit@3 | MRR |
| --- | ---: | ---: | ---: |
| BM25 | 0.72 | 0.89 | 0.818 |
| 混合召回（BM25 + 向量 RRF） | 0.67 | 0.93 | 0.755 |
| 混合 + LLM 重排 | 0.74 | 0.97 | 0.821 |
| 混合 + 查询改写 + 重排 | 0.75 | 0.96 | 0.848 |

补充：语义改写子集 hit@1 从 0.25 → 0.62；生成质量 LLM-as-judge 忠实度 100 / 相关性 99 / 引用 97；RAGAS 忠实度 1.000、答案相关性 0.975、上下文精度 0.550、上下文召回 0.965。

> 数字本身不是重点，重点是**同一套题、一条命令、结论可复现**，并且知道每个配置为什么涨、涨在哪个子集。

## 4. 技术栈

Python 3.13 · FastAPI · OpenAI 兼容 SDK（DeepSeek）· fastembed（bge-small-zh）· rank_bm25 · FAISS · MySQL / SQLite · pytest（70 个用例）· Docker · GitHub Actions

## 5. 怎么验证

```bash
python -m pytest -q                 # 70 个用例，全部不花钱（大模型被 monkeypatch）
python scripts/eval_all.py --large  # 一条命令跑评估，生成 docs/EVAL_REPORT.md
docker compose up -d --build        # 起服务：http://127.0.0.1:8010/docs
```

线上演示：<http://8.137.173.141:8010>（口令见简历，每日有额度上限）

## 6. 还没做的

- 会话记忆是进程内存态，多实例部署要换 Redis；
- 文档增量索引没做（目前上传即全量重建 BM25 词典）；
- 多模态文档（扫描件 OCR）暂不支持，PDF 只抽文本层。

## 7. 学到了什么

检索质量不能靠感觉调：先做评估集，再改参数，每一步记录 hit@1 / MRR，才知道「改写」对语义问法有效、「重排」主要修排序而不是召回，混合召回单独看反而掉分——这些都要靠数据说话。工程侧同理：降级路径、限流、预算、指标、注入防护，都是「能演示」和「能上线」之间的差距。
