# 架构说明

> 面试现场可以按「一张图 → 一条链路 → 三个取舍」来讲：图是下面这些 Mermaid，链路是 `/api/ask/stream`，取舍见文末。

## 1. 组件与边界

```mermaid
flowchart LR
    subgraph 浏览器
        UI[单页前端<br/>static/app.js]
    end

    subgraph "FastAPI 应用（app/）"
        API[API 层<br/>main.py]
        GUARD[访问保护<br/>guard.py]
        RAG[RAG 编排<br/>rag.py]
        SAFE[注入防护<br/>safety.py]
        RET[混合召回<br/>retrieval.py]
        RW[查询改写<br/>rewrite.py]
        RR[重排<br/>rerank.py]
        EMB[向量化<br/>embeddings.py]
        VS[向量索引<br/>vectorstore.py]
        CACHE[结果缓存<br/>cache.py]
        SES[会话记忆<br/>session.py]
        MET[指标/看板<br/>metrics.py]
    end

    subgraph 存储
        MYSQL[(MySQL<br/>documents / chunks)]
        JSON[(data/db.json<br/>降级后端)]
        SQLITE[(SQLite<br/>缓存)]
    end

    LLM[DeepSeek API<br/>改写 / 重排 / 生成]

    UI -->|HTTP / SSE| API
    API --> GUARD --> RAG
    RAG --> SAFE
    RAG --> RW --> LLM
    RAG --> RET
    RET --> EMB --> VS
    RET -->|BM25| RET
    RAG --> RR --> LLM
    RAG --> CACHE --> SQLITE
    RAG --> LLM
    API --> SES
    API --> MET
    API --> MYSQL
    MYSQL -.连不上.-> JSON
```

要点：

- **API 层只做参数校验和错误映射**，编排全在 `rag.py`，所以「离线评估脚本」和「线上接口」跑的是同一条检索链路（`scripts/evaluate.py` 直接调 `rag.retrieve`）。
- **存储可降级**：MySQL 连不上自动回退 `data/db.json`，演示环境不会因为数据库挂了而白屏。
- **向量索引可降级**：有 `faiss` 用 HNSW，没有就用 numpy 精确检索；`RAG_VECTOR_BACKEND=auto` 自动选。

## 2. 一次流式问答的时序

```mermaid
sequenceDiagram
    autonumber
    participant B as 浏览器
    participant A as FastAPI
    participant G as guard
    participant R as rag
    participant V as 向量索引 + BM25
    participant L as DeepSeek
    participant M as metrics

    B->>A: POST /api/ask/stream {question, session_id}
    A->>G: 限流 + 每日预算
    A->>R: answer_stream(question, chunks, history)
    R-->>B: stage「正在改写问题并检索知识库」
    R->>L: 查询改写（可关，失败自动降级）
    R->>V: BM25 召回 + 向量召回
    V-->>R: 两路候选
    R->>R: RRF 融合 → 取 top-N
    R->>L: LLM 重排（可关，失败保持原序）
    R-->>B: trace（各阶段耗时/候选数）
    R-->>B: stage「已找到 N 个片段，正在生成回答」
    R->>R: safety.neutralize + 不可信数据边界
    R->>L: chat_stream(prompt)
    loop 每个 token 块
        L-->>R: delta
        R-->>B: delta
    end
    R->>M: 计数、延迟、token、成本
    R-->>B: done（完整回答 + 引用 + 耗时 + usage）
    A->>R: 写会话记忆（session_id）
```

失败路径都做了降级：改写失败用原问题、重排失败保持 RRF 顺序、大模型报错在 SSE 里发 `error` 事件、前端能拿到可读 `detail`。

## 3. 检索链路的数据形状

```mermaid
flowchart TD
    Q[用户问题] --> QW[改写：原问题 + 2 个同义问法]
    QW --> BM[BM25 关键词召回<br/>CANDIDATE_K=12]
    QW --> VE[向量召回<br/>bge-small-zh + HNSW<br/>CANDIDATE_K=12]
    BM --> RRF[RRF 融合<br/>score = Σ 1/(k+rank), k=60]
    VE --> RRF
    RRF --> RR[LLM 重排 → 取 TOP_K=4]
    RR --> CTX[片段 + 文档名 + 编号]
    CTX --> P[Prompt：规则 + 不可信资料边界 + 问题]
    P --> ANS[带编号引用的回答]
```

## 4. 数据库模型

```mermaid
erDiagram
    documents ||--o{ chunks : "1 篇文档 → N 个切片"
    documents {
        bigint id PK
        varchar name
        varchar collection "多知识库隔离"
        int chunks
        int chars
        timestamp created_at
    }
    chunks {
        bigint id PK
        bigint doc_id FK
        int idx
        mediumtext text
        int chars
    }
```

## 5. 三个关键取舍（面试可直接讲）

| 决策 | 选了什么 | 为什么 | 代价 |
| --- | --- | --- | --- |
| 召回 | BM25 + 向量混合（RRF） | 单路都有明显短板：BM25 不懂同义词，向量对数字/型号/代码标识符弱；两路融合在 100 题上 hit@1 从 0.67 提到 0.74（加重排后 0.75） | 延迟从 3ms 涨到 ~2.6s；候选数要调 |
| 重排 | LLM 打分（可换 cross-encoder） | 少一个模型下载、少一份内存；失败时保持原序不影响可用性 | 每题多一次 API 调用，钱和延迟都涨 |
| 会话 | 服务端 session_id + 进程内 TTL 存储 | 客户端传 history 可以伪造、且长对话越传越大；服务端存最近 6 轮，注入时再截断 | 单机内存态，重启丢失；多实例要换 Redis |

另外两个小取舍：

- **注入防护只中和「硬模式」**：知识库里讨论提示注入的文章不该被删改，真正拦住注入的是「不可信数据边界 + 系统规则」。
- **指标自己写，不引 prometheus_client**：只需要计数、分位数、求和三件事，少一个依赖；暴露 `/api/metrics`（Prometheus 文本）+ `/api/dashboard`（JSON 给前端展示）。
