# 📚 RAG 知识库问答

> [English version](README.en.md)

一个**开箱即用**的 RAG（检索增强生成）知识库问答系统：上传文档 → 自动切分 → 混合检索（BM25 + 向量）→ 查询改写 + 重排 → 让大模型基于资料回答，并标注引用来源。

- 纯 Python，**不需要 GPU**，不需要独立向量数据库，`pip install` 五分钟跑起来
- 检索链路可拆可换：BM25 / 向量检索 / RRF 融合 / LLM 重排 / 查询改写，每一层都能单独关掉做对比
- 前端 + 后端 + 测试 + CI 齐全，适合作为**实习/求职项目**
- 中文场景优化：按句切分、中文分词、bge 中文向量模型、引用可追溯

[![CI](https://github.com/HDC233666/rag-knowledge-base/actions/workflows/ci.yml/badge.svg)](https://github.com/HDC233666/rag-knowledge-base/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-yellow)

| 对话界面（回答 + 引用） | API 文档（Swagger UI） |
|---|---|
| ![对话界面](docs/screenshots/chat.png) | ![API 文档](docs/screenshots/api.png) |

---

## 功能

| 功能 | 说明 |
|---|---|
| 文档上传 | 支持 txt / md / pdf / csv / log，单文件 ≤ 20MB |
| 自动切分 | 按句切分 + 目标字数 + 重叠，避免语义被切断 |
| 混合检索 | BM25（词频）+ 向量检索（bge-small-zh，CPU 可跑），RRF 融合两路排名 |
| 向量索引 | 默认 numpy 精确检索（零依赖），装了 faiss 自动切到 HNSW 近似检索；`RAG_VECTOR_BACKEND=numpy/faiss/auto` 可强制 |
| 查询改写 | 让大模型把一个问题改写成多个检索式（多查询召回），提升召回率 |
| 重排 | 对候选片段重排：LLM 打分（默认）或 bge-reranker 交叉编码器（本地 CPU，几十毫秒） |
| 结果缓存 | 改写/重排结果落 SQLite + TTL，重复问题从「秒级」降到「毫秒级」 |
| 回答 + 引用 | 回答里带 `[1][2]` 编号，界面展示每条引用的来源片段和得分 |
| 防幻觉 | 提示词强约束「资料里没有就说没有」；检索为空时直接拒答 |
| 持久化 | 默认存 `data/db.json`，重启不丢；配置 MySQL 后存数据库（见「接入 MySQL」） |
| MySQL 存储 | 可选：文档 + 切分块入库，演示表设计、索引、外键级联；连不上自动降级 JSON |
| 检索评估 | 内置 100 题评估集 + 脚本，一键对比 BM25 / 混合 / 重排 / 改写的 hit@1、hit@3、MRR |
| 生成质量评估 | 跑通「生成 → LLM 裁判打分」流程，输出忠实度 / 相关性 / 引用质量，思路对齐 RAGAS |
| RAGAS 四指标 | 忠实度 / 答案相关性 / 上下文精度 / 上下文召回，20 题子集，脚本 `scripts/ragas_eval.py` |
| 真实语料评估 | 抓取 114 篇 FastAPI 中文文档（2100 切片）作干扰语料，对比 14 切片 vs 千级切片的检索质量 |
| 流式回答 | `/api/ask/stream` 走 SSE：先推检索结果，再逐字推回答；界面边生成边显示 |
| 多知识库 | 每篇文档属于一个 collection（默认 `default`），侧栏可切换 / 新建；不同知识库的文档与检索互不干扰 |
| 答案反馈 | 回答下方 👍/👎，写入 `data/feedback.jsonl`，`/api/feedback/stats` 看统计 |
| 可观测 | `/api/metrics` 暴露 Prometheus 指标（提问数、错误数、延迟、流式数） |
| 多轮对话 | 前端自动带上最近几轮问答，追问（「那向量检索的缺点呢？」）能接上文 |
| 限流 | 每 IP 每分钟请求数上限（`RAG_RATE_LIMIT`），超限返回 429，防止接口被刷 |
| 界面 | 原生 HTML/CSS/JS，无构建工具，支持深色模式，回答自动渲染 Markdown |
| 测试 | pytest，51 个用例覆盖切分/检索/融合/重排/缓存/向量索引/多知识库/反馈/指标/接口/流式/限流；GitHub Actions 自动跑 |

## 架构

```mermaid
flowchart LR
    A[用户提问] --> B[FastAPI /api/ask]
    B --> R[查询改写<br/>LLM 生成多个检索式]
    R --> C1[BM25 召回<br/>稀疏检索]
    R --> C2[向量召回<br/>bge-small-zh]
    C1 --> F[RRF 融合<br/>多路排名合并]
    C2 --> F
    F --> G[LLM 重排<br/>top-k 精排]
    G --> D[拼上下文<br/>参考资料]
    D --> E[DeepSeek 大模型]
    E --> H[回答 + 引用 + trace]
    U[上传文档] --> I[文本抽取<br/>pdf/md/txt]
    I --> J[切分 chunk<br/>重叠]
    J --> K[存入 db.json<br/>或 MySQL]
    K --> C1
    K --> C2
```

> 任一环节都可以用 `.env` 关掉（`RAG_ENABLE_REWRITE=0` / `RAG_ENABLE_RERANK=0`），关掉后自动降级，方便做消融对比。

## 快速开始

**环境要求**：Python 3.11+（Windows / macOS / Linux 均可）

```bash
# 1. 进入项目目录
cd rag-knowledge-base

# 2. 创建虚拟环境（推荐，避免污染全局 Python）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 Key：复制 .env.example 为 .env，填入你的 DeepSeek API Key
#    （如果系统里已经设置过 DEEPSEEK_API_KEY 环境变量，可跳过）
copy .env.example .env
notepad .env

# 5. 启动
uvicorn app.main:app --reload
```

浏览器打开 **http://127.0.0.1:8000** 即可。

> 第一次启动会自动把 `data/sample/` 里的示例文档导入，可以直接问「什么是 RAG？」试试。
> 接口文档：http://127.0.0.1:8000/docs

### 配置项（.env）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API Key（也可用系统环境变量） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名，可换 `deepseek-reasoner` 等 |
| `RAG_TOP_K` | `4` | 每次检索喂给模型的片段数 |
| `RAG_CHUNK_SIZE` | `600` | 每块目标字数 |
| `RAG_CHUNK_OVERLAP` | `80` | 相邻块重叠字数 |
| `RAG_TEMPERATURE` | `0.2` | 回答温度 |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 向量模型（CPU 可跑，512 维）；换模型后向量自动重建 |
| `RAG_EMBED_BACKEND` | `auto` | 设 `hash` 强制用零依赖哈希向量（CI/无网环境用） |
| `RAG_EMBED_QUERY_PREFIX` | bge 查询指令 | bge 系列查询侧建议加的指令前缀，换模型可留空 |
| `RAG_VECTOR_BACKEND` | `auto` | 向量索引后端：`auto`（有 faiss 用 HNSW）/ `numpy`（精确）/ `faiss`（近似） |
| `RAG_FAISS_M` | `32` | HNSW 每个节点的连边数：越大越准、越占内存 |
| `RAG_FAISS_EF_SEARCH` | `64` | HNSW 查询时搜索的候选数：越大越准、越慢 |
| `RAG_CANDIDATE_K` | `12` | BM25 与向量各召回多少候选，再融合 |
| `RAG_RRF_K` | `60` | RRF 融合常数（论文经验值） |
| `RAG_ENABLE_REWRITE` | `1` | 查询改写开关；`0` = 省一次大模型调用 |
| `RAG_ENABLE_RERANK` | `1` | 重排开关；`0` = 省一次重排开销 |
| `RAG_RERANK_BACKEND` | `llm` | 重排后端：`llm`（LLM 打分）或 `cross`（bge-reranker 交叉编码器，本地 CPU） |
| `RAG_CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-base` | `cross` 后端用的模型 |
| `RAG_ENABLE_CACHE` | `1` | 改写/重排结果缓存开关（SQLite + TTL） |
| `RAG_CACHE_TTL` | `3600` | 缓存有效期（秒） |
| `RAG_CACHE_PATH` | `data/cache.db` | 缓存文件位置（已 gitignore） |
| `RAG_RATE_LIMIT` | `60` | 每 IP 每分钟问答次数上限，`0` = 关闭 |
| `RAG_DB_HOST` | 空 | 填了就用 MySQL；留空用 JSON 文件 |
| `RAG_DB_PORT` | `3306` | MySQL 端口 |
| `RAG_DB_USER` | `root` | 数据库账号 |
| `RAG_DB_PASSWORD` | 空 | 数据库密码（只写进 `.env`，别提交） |
| `RAG_DB_NAME` | `rag_kb` | 数据库名 |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（含模型名、Key 是否配置、向量后端） |
| GET | `/api/stats?collection=` | 知识库统计（文档数 / 块数 / 字数） |
| GET | `/api/collections` | 知识库列表（每个库的文档数与块数） |
| GET | `/api/documents?collection=` | 文档列表 |
| POST | `/api/documents` | 创建文档（JSON：`{name, text, collection?}`） |
| POST | `/api/upload` | 上传文件（multipart，可带 `collection` 字段） |
| DELETE | `/api/documents/{id}` | 删除文档 |
| POST | `/api/ask` | 提问（JSON：`{question, top_k?, history?, collection?}`）→ 一次性返回 |
| POST | `/api/retrieve` | **只检索不生成**（压测/评估用，默认 `llm=false` 不花 API 费用）：`{question, collection?, top_k?, llm?}` → `{hits, trace, latency_ms}` |
| POST | `/api/ask/stream` | 同上入参，**SSE 流式**：`trace` / `delta` / `done` 事件 |
| POST | `/api/feedback` | 答案反馈 `{rating: up\|down, question?, comment?, collection?}` |
| GET | `/api/feedback/stats` | 反馈统计（👍/👎 数量 + 最近 5 条） |
| GET | `/api/metrics` | Prometheus 文本指标 |

流式接口的事件（每条 `data: {...}`）：

| 事件 | 载荷 | 含义 |
|---|---|---|
| `trace` | `{queries, rewrite, rerank, candidates, citations}` | 检索完成，先把引用与 trace 推给前端 |
| `delta` | `{text}` | 回答增量文本（逐字显示） |
| `done` | `{answer, citations, latency_ms, tokens, trace}` | 整轮结束 |
| `error` | `{detail}` | 出错 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"什么是 RAG？\"}"
```

限流：默认每个 IP 每分钟 60 次（`RAG_RATE_LIMIT`，设为 0 关闭）。超限返回 `429 {"detail": "请求太频繁了，稍后再试"}`，`/api/health` 不受限。

## 项目结构

```
rag-knowledge-base/
├── app/
│   ├── main.py        # FastAPI 入口 + 路由
│   ├── config.py      # 配置（环境变量 / .env）
│   ├── loaders.py     # 文件 → 文本（pdf/md/txt）
│   ├── chunking.py    # 文本切分（句子 + 重叠）
│   ├── embeddings.py  # 向量模型（bge-small-zh / 哈希降级）
│   ├── vectorstore.py # 向量索引（numpy 精确 / FAISS HNSW 近似，自动降级）
│   ├── retrieval.py   # BM25 + 向量混合检索（RRF 融合）
│   ├── rewrite.py     # 查询改写（多查询）
│   ├── rerank.py      # LLM 重排
│   ├── store.py       # 文档仓库（JSON 实现，默认）
│   ├── db.py          # MySQL 存储实现 + 自动降级
│   ├── llm.py         # 调用 DeepSeek（重试 + 报错）
│   └── rag.py         # RAG 主流程（改写→混合召回→融合→重排→生成）
├── static/            # 前端（无构建工具）
├── data/
│   ├── sample/        # 示例知识库（含干扰文档，评估用）
│   ├── eval/          # 评估集（questions.json）与报告（report.json）
│   └── db.json        # 运行时生成（已 gitignore；用 MySQL 时不生成）
├── scripts/
│   ├── evaluate.py    # 检索评估：hit@1 / hit@3 / MRR 多配置对比（100 题）
│   ├── corpus_fastapi.py  # 抓取 FastAPI 中文文档（真实语料扩容）
│   ├── ingest_corpus.py   # 把抓取语料导入独立集合 fastapi-docs
│   ├── ragas_eval.py      # RAGAS 四指标（忠实度/相关性/上下文精度/召回）
│   ├── evaluate_generation.py  # 生成质量评估：LLM 裁判打忠实度/相关性/引用分
│   └── smoke.py       # 真实接口冒烟
├── docs/
│   └── LEARNING-RAG-ADVANCED.md  # 面向零基础的进阶讲解（原理 + 评估 + 面试问答）
├── tests/             # pytest 测试（39 个用例）
└── .github/workflows/ci.yml
```

## 运行测试

```bash
pytest -q
```

## 工程指标（本机压测）

| 指标 | 实测 | 说明 |
|---|---|---|
| 检索延迟（单并发） | p50 **44 ms** / p95 **48 ms** | 14 切片：bge 查询向量 + BM25 + RRF |
| 检索吞吐（5 并发） | **~25 QPS**，p95 **~200 ms** | 单进程 uvicorn + ONNX CPU |
| 检索吞吐（20 并发） | **~31 QPS**，p95 **~700 ms** | 并发上来后 embedding 排队 |
| 端到端回答 | **1.2 – 2.6 s** | 检索 + Prompt + LLM + 引用 |
| 每次问答成本 | **≈ ¥0.003** | ~1500 tokens；1000 次 ≈ 3 元 |
| 冷启动 | **~8 s** | 加载 bge-small-zh（95MB ONNX） |

```bash
# 只压检索链路，不花 API 费用
python scripts/bench.py --path /api/retrieve --total 60 --concurrency 5
# 压 Web 层开销
python scripts/bench.py --path /api/stats --total 200 --concurrency 50
```

> 单进程 + CPU ONNX 是吞吐瓶颈；生产做法是 uvicorn 多 worker 或把 embedding 拆成独立服务。
> 查询向量做了进程内 LRU 缓存，重复问题直接跳过 embedding。

## 检索评估

内置 100 道题（基础 20 + 语义改写 8 + 进阶 34 + 章节/干扰 38）和 `scripts/evaluate.py`，一次跑四种配置，直接对比检索质量：

```bash
python scripts/evaluate.py          # 100 题，需要 .env 里配好 Key（重排/改写要用大模型）
python scripts/evaluate.py --no-llm # 只对比 BM25 与混合检索，不花钱
```

指标说明：**hit@1** = 正确片段排第 1 的比例；**hit@3** = 排进前 3 的比例；**MRR** = 正确片段排名的倒数均值（越接近 1 越好）。

当前基线（14 个切片、100 道题、bge-small-zh）：

| 配置 | hit@1 | hit@3 | MRR | 平均延迟 |
|---|---|---|---|---|
| 纯 BM25 | **0.72** | 0.93 | 0.818 | 3 ms |
| 混合召回（BM25 + 向量，RRF） | 0.67 | 0.91 | 0.790 | 14 ms |
| 混合召回 + LLM 重排 | 0.74 | 0.94 | 0.838 | 1.49 s |
| 混合召回 + 查询改写 + 重排 | **0.75** | **0.96** | **0.848** | 2.64 s |

按题型拆开看，才知道每种改动到底在帮谁（hit@1）：

| 题目分组 | 题数 | 纯 BM25 | 混合召回 | 混合 + 重排 |
|---|---|---|---|---|
| 基础概念 q01–q20 | 20 | 0.70 | 0.65 | **0.80** |
| 语义改写 q21–q28（口语化 / 换说法） | 8 | 0.25 | **0.50** | **0.62** |
| 进阶原理 q29–q62 | 34 | **0.74** | 0.71 | 0.71 |
| 章节 / 干扰 q63–q100 | 38 | **0.82** | 0.68 | 0.76 |

> 这张表比总分更值得讲：**语义改写题上 BM25 只有 0.25，混合召回直接翻倍到 0.50，重排再到 0.62**——这就是「向量检索 + 重排」真正解决的问题；反过来，章节题的问题大多直接抄了文档标题/关键词，BM25 近似「开卷抄答案」（0.82），语义化改动反而稀释字面优势。
> 所以别只报总分：小语料上总分会掩盖结构差异，评估必须分题型看。代价也看得见——延迟从 3 ms 涨到 2.6 s，因为重排和改写都要调大模型；生产上用缓存把「贵的东西只算一次」（见下）。

### 真实语料评估（千级切片）

`scripts/corpus_fastapi.py` 抓取 FastAPI 官方中文文档 114 篇（约 1 MB → 2100 个切片），经 `scripts/ingest_corpus.py` 入库为独立知识库 `fastapi-docs`；评估时用 `--extra-corpus` 把这批切片作为**干扰文档**混入检索池，模拟真实企业知识库的规模：

```bash
python scripts/corpus_fastapi.py --limit 120       # 抓取公开文档
python scripts/ingest_corpus.py                    # 入库到 fastapi-docs 集合
python scripts/evaluate.py --extra-corpus data/corpus/fastapi --report-suffix _large
```

| 语料规模 | 配置 | hit@1 | hit@3 | MRR | 平均延迟 |
|---|---|---|---|---|---|
| 14 切片 | 纯 BM25 | 0.72 | 0.93 | 0.818 | 3 ms |
| 2114 切片 | 纯 BM25 | 0.61 | 0.87 | 0.733 | 181 ms |
| 2114 切片 | 混合召回 | 0.57 | 0.83 | 0.688 | 939 ms |
| 2114 切片 | 混合 + 重排 | **0.76** | 0.93 | 0.842 | 1.5 s |
| 2114 切片 | 混合 + 改写 + 重排 | 0.75 | **0.96** | **0.852** | 2.7 s |

> 规模一上来，单路召回集体掉点（BM25 0.72 → 0.61，混合 0.67 → 0.57）：干扰片段变多，字面命中被淹没；**重排把 hit@1 拉回 0.76**，这就是「粗排 + 精排」两阶段检索在真实规模下的价值。
> 延迟也值得讲：BM25 从 3 ms 涨到 181 ms（要扫 2114 个切片）、混合 939 ms（向量矩阵 + 词法双路），生产上要靠分片、倒排索引和缓存压下来——这比「小语料上跑个 demo」更接近真实工程。

### 缓存效果

改写/重排结果按「问题 + 候选指纹」缓存在 SQLite（`RAG_CACHE_TTL`，默认 1 小时）。同一批问题重复请求：

| 场景 | 平均耗时 |
|---|---|
| 首次（冷启动，改写 + 重排各一次 LLM 调用） | 1.4 ~ 2.6 s |
| 命中缓存（改写、重排分数都直接读 SQLite） | **28 ms**（约 50 倍） |

> 生产里这就是「把贵的东西算一次」：模型没变、语料没变，答案就该能复用。

### RAGAS 四指标（20 题）

`scripts/ragas_eval.py` 按 RAGAS 的四个核心维度做「LLM 当裁判」评估（20 题，生成 + 4 次裁判调用）：

| 指标 | 含义 | 得分（20 题，0–1） |
|---|---|---|
| 忠实度 faithfulness | 答案里的每个说法能否在检索片段中找到依据 | **1.000** |
| 答案相关性 answer_relevancy | 答案是否切题（0–10 归一化） | **0.975** |
| 上下文精度 context_precision | 检索到的片段里真正有用的比例 | 0.550 |
| 上下文召回 context_recall | 标注的目标片段信息是否被检索结果覆盖 | **0.965** |

> 两个诚实的说明：
> 1. 上下文精度 0.55 偏低是**结构性的**：top_k=4 里通常只有 1 个目标片段，理论上限就是 0.25–0.5；它衡量的是「召回的所有片段里有用占比」，不是命中率；
> 2. 裁判和被评测模型是同一个，绝对分数偏宽容，看趋势和横向对比即可（换模型 / 改 prompt / 开不开重排）。

### 重排后端对比

| 后端 | 说明 | 延迟 | 依赖 |
|---|---|---|---|
| `llm`（默认） | 让大模型给候选打 0~10 分 | ~1.4 s | 需要 API Key |
| `cross` | bge-reranker 交叉编码器（fastembed/ONNX 本地推理） | 几十 ms | 首次需下载模型（约 100 MB） |

`cross` 模式如果模型下载失败或推理报错，会自动回退到 `llm` 重排，链路不会断。

### 生成质量评估（LLM-as-a-judge）

检索指标只衡量「片段找没找到」，用户看到的却是「答案好不好」。`scripts/evaluate_generation.py` 对每道题真实走一遍 RAG 流程，再让裁判模型按 RAGAS 思路打分：

```bash
python scripts/evaluate_generation.py --limit 20   # 20 题，每题 2 次大模型调用（生成 + 打分）
```

| 指标 | 含义 | 得分（20 题，0-100） |
|---|---|---|
| 忠实度 faithfulness | 回答里的结论能否在引用原文中找到依据（防幻觉） | **100** |
| 答案相关性 relevance | 是否正面、完整地回答了问题 | **99** |
| 引用质量 citation | 引用的片段是否真的支撑回答 | **97** |
| 拒答 | 资料确实没讲时，是否诚实说「找不到」 | 1 / 20 |

> 两个诚实的说明：
> 1. 裁判拿到的是**检索到的完整原文**（不是界面展示用的 160 字摘要），否则会把「摘要没覆盖」误判成「模型编造」——第一版就踩了这个坑，分数只有 65/75/63；
> 2. 裁判和被评测的是**同一个模型**，会偏宽容，绝对分数只能看趋势，价值在横向对比（改 prompt / 换模型 / 开不开重排）。生产里应换更强的裁判模型并做人工抽检。

### 向量索引后端（numpy / FAISS）

| 后端 | 适用规模 | 特点 |
|---|---|---|
| `numpy`（默认兜底） | 几千块以内 | 矩阵乘一次算精确余弦，零依赖、结果精确 |
| `faiss`（HNSW） | 万级以上 | 近似最近邻，查询接近对数复杂度；`M` 控制连边数、`efSearch` 控制搜索宽度，越大越准越慢 |

`RAG_VECTOR_BACKEND=auto` 时：装了 faiss 就用 HNSW，没装或建索引失败自动退回 numpy，服务不受影响。本项目的示例语料只有 14 块，两者延迟差异可以忽略——**FAISS 的收益在规模**，这里把它做出来是为了讲清「什么时候该换索引」，而不是为了微秒级提升。

## 进阶：接入 MySQL（可选）

默认使用 JSON 文件存储；需要数据库存储时，可切换为 MySQL（无需改代码）：

**1. 建库建账号（一次即可）**

```sql
CREATE DATABASE rag_kb DEFAULT CHARACTER SET utf8mb4;
CREATE USER 'dsh_app'@'localhost' IDENTIFIED BY '你的密码';
GRANT ALL PRIVILEGES ON rag_kb.* TO 'dsh_app'@'localhost';
```

**2. 在 `.env` 里填数据库连接**

```ini
RAG_DB_HOST=127.0.0.1
RAG_DB_PORT=3306
RAG_DB_USER=dsh_app
RAG_DB_PASSWORD=你的密码
RAG_DB_NAME=rag_kb
```

**3. 重启服务**，`/api/stats` 返回 `"storage": "mysql"` 就成功了（表会自动创建）。用 DataGrip / Navicat 打开 `rag_kb` 就能看到数据。

**表结构**（建表 SQL 在 `app/db.py`）：

| 表 | 字段 | 说明 |
|---|---|---|
| `documents` | id, name, chunks, chars, created_at | 一篇文档一行 |
| `chunks` | id, doc_id, idx, text, chars | 一个切分块一行，`doc_id` 外键关联 `documents` |

**设计要点：**

- **外键 + 级联删除**：`ON DELETE CASCADE` 保证删文档时自动清掉它的所有块，不留孤儿数据；
- **索引**：`doc_id` 上建索引，按文档查块、级联删除都不全表扫描；
- **降级**：数据库连不上时自动退回 JSON 存储（`make_store()`），保证 demo 在任何电脑都能跑；
- **取舍**：目前每次操作新建连接 + autocommit（教学演示够用），生产应换连接池，并给 `text` 加全文索引或接向量库。

## 设计说明

**为什么从 BM25 升级到混合检索（而不是一把梭上向量）？**

- BM25 是词面匹配，毫秒级、零依赖，在「问题里带着原文关键词」时非常准；
- 向量检索补的是「换个说法」：用户说「资料更新了模型还在答旧的」，文档里写的是「外挂知识库能随时更新」，词面不重合，向量能对上；
- 两路各召回 `RAG_CANDIDATE_K` 个候选，用 **RRF（Reciprocal Rank Fusion）** 按排名融合——它只看名次不看分数，天然避免「BM25 分和余弦相似度量纲不同」的问题，`score = Σ 1/(k + rank)`；
- 向量模型选 `bge-small-zh-v1.5`：512 维、CPU 可跑、中文效果好，不需要 GPU 和独立向量库；查询侧按 bge 官方建议加指令前缀，文档侧不加。

**为什么加 LLM 重排和查询改写？**

- 重排：召回阶段追求「别漏」（top-12 宽召回），重排阶段追求「排得准」（LLM 给候选打相关性分，取前 4）；
- 改写：用户问题往往口语化、信息少，让大模型改写成 2–3 个检索式分别召回再融合，能明显提升命中率（语义改写题上 hit@1 从 0.50 → 0.62，整体 0.74 → 0.75）；
- 两者都靠大模型，代价是延迟（毫秒级 → 秒级），所以做成开关，可评估、可关闭。

**为什么把评估集扩到 100 题、还要分题型看？**

- 评估集从 62 扩到 100 后，加入了更多原理题和干扰文档题，题目构成变了，指标也会整体漂移——这不是模型变强/变弱，而是**考卷换了**，报数必须注明题数与构成；
- 只看总分会出现「纯 BM25 赢混合检索」的错觉（0.72 vs 0.67）：因为题目大量直接取自文档关键词，字面匹配近乎作弊；
- 按题型拆开后结论才清楚：**语义改写题 BM25 0.25 → 混合 0.50 → 重排 0.62**，而章节题 BM25 0.82 基本吃满。面试时讲「我怎么发现评估集偏置、怎么分层归因」，比背一个漂亮总分更有说服力。

**为什么加 FAISS 向量索引？**

- numpy 精确检索是 O(N·D)：几万块以后每次提问都要把整库过一遍，延迟随规模线性涨；
- FAISS HNSW 用多层图做近似最近邻，查询接近对数复杂度，代价是召回不是 100%（可调 `efSearch` 换精度）；
- 做成 `auto` 后端 + 失败自动退回 numpy：既能在小库上保持零依赖，也能在大库上演示「规模上来该换什么」。同一条检索接口下两种实现可切换，是这类项目最值得展示的「可演进架构」。

**为什么加流式、多轮与限流？**

- 流式：检索 + 生成要 1–2 秒，全量返回前界面是空白的；SSE 先把检索 trace 和引用推给前端，再逐字推回答，等待感明显下降；
- 多轮：追问（「那它的缺点呢？」）不带上下文会被当成新问题，前端带最近 6 条消息后体验才像聊天助手；
- 限流：问答要花 token，公开 demo 被刷会烧钱；按 IP 每分钟限 60 次，超限 429，健康检查放行。

**为什么加多知识库（collection）？**

- 一个知识库装所有资料会互相污染：问「公司报销流程」时，旅行攻略或个人笔记也可能被当成候选片段；
- 做法是每篇文档带 `collection` 字段（默认 `default`），检索只在该库的切片里做，等于给混合检索加了「数据隔离」这一层；
- MySQL 用 `collection` 列 + 索引过滤，JSON 后端按字段过滤；前端侧栏可切换与新建，切库时文档列表、统计、问答全部跟着切；
- 面试可以讲清楚边界：这是**逻辑隔离**不是权限系统——真做多租户还要加用户身份、库级鉴权与配额。

**降级策略（面试常问）**

- 向量模型加载失败 → 自动退化为哈希向量（`HashEmbedder`），保证服务不挂；
- 某一路检索异常 → 退化为纯 BM25；
- 重排/改写输出不是合法 JSON → 保留原顺序（重排）或退回原问题（改写）；
- MySQL 连不上 → 退化为 JSON 存储。

正确性靠测试保证：`tests/test_hybrid.py`、`tests/test_pipeline.py` 覆盖了融合排序、坏输出兜底、trace 字段等边界。

**如何扩展到生产可用？**

1. 存储：JSON（默认）→ MySQL（已支持）→ 向量库（Chroma / Milvus）；
2. 检索：混合检索（已支持）→ 向量库 HNSW 索引 + 多路召回；
3. 切分：按格式（Markdown 标题、PDF 版面）结构化切分；
4. 评估：已扩到 100 题并做分题型分析、LLM 裁判生成评估；再进一步可接 RAGAS 官方指标、建人工抽检集；
5. 工程：权限控制、日志与监控、重排缓存（已支持）、多轮对话（已支持）、限流（已支持）。

## Roadmap

- [x] MySQL 存储后端（documents / chunks 两张表）
- [x] 向量检索（bge-small-zh）+ BM25 混合检索（RRF 融合）
- [x] LLM 重排 + 查询改写
- [x] 检索评估（62 题，hit@1 / hit@3 / MRR，多配置对比）
- [x] 流式回答（SSE，检索 trace + 回答逐字）
- [x] 多轮对话上下文（前端携带最近几轮）
- [x] 每 IP 限流（`RAG_RATE_LIMIT`）
- [x] 向量索引后端：numpy 精确 / FAISS HNSW 近似（自动降级）
- [x] 评估集扩到 100 题 + 分题型归因
- [x] 生成质量评估（忠实度 / 相关性 / 引用质量，LLM-as-a-judge）
- [x] 多知识库（collection 隔离，前端可切换/新建）
- [x] 答案评分/反馈按钮（👍/👎 + 统计接口）
- [x] Prometheus 指标（`/api/metrics`）
- [ ] Docker 部署

## 安全提示

`.env`（含 API Key 与数据库密码）和 `data/db.json` 已在 `.gitignore` 中忽略，不会提交到仓库；本 README 与示例配置里也不含任何真实密钥。

## 常见问题

- **报错 `没有找到 DEEPSEEK_API_KEY`**：检查 `.env` 是否在项目根目录、是否填了值；或设置系统环境变量后重启终端。
- **接口报 404 / model not found**：`.env` 里的 `DEEPSEEK_MODEL` 换回 `deepseek-chat`。
- **回答「知识库里没有找到」**：先确认侧边栏有文档；换了说法的问题一般也能被混合检索找到，若仍不行，换个更贴近原文的关键词再问。
- **PDF 抽不出文字**：扫描版 PDF 是图片，需要 OCR（本项目暂不支持）。

## 许可证

[MIT](LICENSE)
