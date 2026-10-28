# 📚 RAG Knowledge Base Q&A

> [中文版](README.md)

An **out-of-the-box** RAG (Retrieval-Augmented Generation) knowledge-base Q&A app: upload documents → auto-chunk → hybrid retrieval (BM25 + embeddings) → query rewriting + reranking → let the LLM answer from your materials, with traceable citations.

- Pure Python — **no GPU**, no standalone vector database, running in 5 minutes with `pip install`
- Every retrieval stage is pluggable: BM25 / vector search / RRF fusion / LLM rerank / query rewriting — each can be switched off for ablation
- Frontend + backend + tests + CI included — a solid **internship / job-hunting project**
- Chinese-first optimizations: sentence-aware chunking, Chinese tokenizer, bge Chinese embeddings, traceable citations

[![CI](https://github.com/HDC233666/rag-knowledge-base/actions/workflows/ci.yml/badge.svg)](https://github.com/HDC233666/rag-knowledge-base/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-yellow)

| Chat UI (answer + citations) | API docs (Swagger UI) |
|---|---|
| ![Chat UI](docs/screenshots/chat.png) | ![API docs](docs/screenshots/api.png) |

---

## Features

| Feature | Description |
|---|---|
| Document upload | txt / md / pdf / csv / log, up to 20 MB per file |
| Auto chunking | Sentence-aware splitting + target size + overlap, so meaning is never cut in half |
| Hybrid retrieval | BM25 (lexical) + vector search (bge-small-zh, CPU-only), fused with RRF |
| Vector index | numpy exact search by default (zero deps); FAISS HNSW kicks in automatically when installed (`RAG_VECTOR_BACKEND=numpy/faiss/auto`) |
| Query rewriting | The LLM expands one question into several retrieval queries (multi-query recall) |
| Reranking | LLM scoring (default) or bge cross-encoder (local CPU) reorders candidates |
| Caching | Rewrite/rerank results cached in SQLite with TTL — repeat questions go from seconds to milliseconds |
| Answers with citations | Answers carry `[1][2]` markers; the UI shows the source snippet and score of each citation |
| Anti-hallucination | Prompt strictly says "if it's not in the materials, say so"; empty retrieval → refuse to answer |
| Persistence | `data/db.json` by default; set MySQL vars to store in a database (see below) |
| MySQL storage | Optional: documents + chunks in MySQL — table design, indexes, cascade delete; falls back to JSON if unreachable |
| Retrieval eval | 100-question eval set + script: one command to compare BM25 / hybrid / rerank / rewrite on hit@1, hit@3, MRR |
| Generation eval | End-to-end generation + LLM-as-a-judge scoring: faithfulness / relevance / citation quality, RAGAS-style |
| Streaming | `/api/ask/stream` over SSE: retrieval trace first, then the answer token by token |
| Multi-turn | The frontend sends recent turns so follow-up questions keep context |
| Collections | Every document belongs to a collection (default `default`); the sidebar switches/creates them, and documents + retrieval stay isolated per collection |
| Rate limiting | Per-IP request cap (`RAG_RATE_LIMIT`), `429` when exceeded |
| UI | Vanilla HTML/CSS/JS, no build step, dark mode supported, Markdown rendering |
| Tests | pytest with 46 cases covering chunking / retrieval / fusion / reranking / caching / API / streaming / rate limiting; GitHub Actions runs them automatically |

## Architecture

```mermaid
flowchart LR
    A[User question] --> B[FastAPI /api/ask]
    B --> R[Query rewriting<br/>LLM generates several queries]
    R --> C1[BM25 recall<br/>lexical retrieval]
    R --> C2[Vector recall<br/>bge-small-zh]
    C1 --> F[RRF fusion<br/>merge ranked lists]
    C2 --> F
    F --> G[LLM rerank<br/>top-k precision]
    G --> D[Build context<br/>reference materials]
    D --> E[DeepSeek LLM]
    E --> H[Answer + citations + trace]
    U[Upload document] --> I[Text extraction<br/>pdf/md/txt]
    I --> J[Chunk text<br/>with overlap]
    J --> K[Store in db.json<br/>or MySQL]
    K --> C1
    K --> C2
```

> Every stage can be disabled via `.env` (`RAG_ENABLE_REWRITE=0` / `RAG_ENABLE_RERANK=0`) and gracefully degrades — handy for ablation studies.

## Quick start

**Requirements**: Python 3.11+ (Windows / macOS / Linux)

```bash
# 1. Enter the project directory
cd rag-knowledge-base

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the key: copy .env.example to .env and fill in your DeepSeek API Key
#    (skip if DEEPSEEK_API_KEY is already set in your environment)
copy .env.example .env
notepad .env

# 5. Run
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**.

> On first start the sample documents in `data/sample/` are imported automatically — ask "什么是 RAG？" (What is RAG?) to try it.
> API docs: http://127.0.0.1:8000/docs

### Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | empty | DeepSeek API Key (system environment variable also works) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name, e.g. `deepseek-reasoner` |
| `RAG_TOP_K` | `4` | Number of chunks fed to the model per question |
| `RAG_CHUNK_SIZE` | `600` | Target characters per chunk |
| `RAG_CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |
| `RAG_TEMPERATURE` | `0.2` | Answer temperature |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding model (CPU-friendly, 512 dims); vectors rebuild automatically when changed |
| `RAG_EMBED_BACKEND` | `auto` | `hash` forces the dependency-free hash embedder (CI / offline) |
| `RAG_EMBED_QUERY_PREFIX` | bge instruction | Query-side instruction prefix recommended for bge models; may be left empty |
| `RAG_VECTOR_BACKEND` | `auto` | Vector index: `auto` (FAISS when available) / `numpy` (exact) / `faiss` (HNSW) |
| `RAG_FAISS_M` | `32` | HNSW links per node: higher = more accurate, more memory |
| `RAG_FAISS_EF_SEARCH` | `64` | HNSW search width: higher = more accurate, slower |
| `RAG_CANDIDATE_K` | `12` | Candidates recalled by BM25 and by vectors before fusion |
| `RAG_RRF_K` | `60` | RRF fusion constant (empirical value from the paper) |
| `RAG_ENABLE_REWRITE` | `1` | Query rewriting on/off; `0` saves one LLM call |
| `RAG_ENABLE_RERANK` | `1` | LLM reranking on/off; `0` saves one LLM call |
| `RAG_RERANK_BACKEND` | `llm` | `llm` (LLM scoring) or `cross` (bge cross-encoder, local CPU) |
| `RAG_CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-base` | Model used by the `cross` backend |
| `RAG_ENABLE_CACHE` | `1` | Cache rewrite/rerank results in SQLite with TTL |
| `RAG_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `RAG_CACHE_PATH` | `data/cache.db` | Cache file location (gitignored) |
| `RAG_RATE_LIMIT` | `60` | Max questions per IP per minute; `0` disables |
| `RAG_DB_HOST` | empty | Set it to use MySQL; empty = JSON file |
| `RAG_DB_PORT` | `3306` | MySQL port |
| `RAG_DB_USER` | `root` | Database user |
| `RAG_DB_PASSWORD` | empty | Database password (keep it in `.env`, never commit) |
| `RAG_DB_NAME` | `rag_kb` | Database name |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check (model name, key configured, vector backend) |
| GET | `/api/stats?collection=` | Knowledge-base stats (documents / chunks / characters) |
| GET | `/api/collections` | List collections (documents + chunks per collection) |
| GET | `/api/documents?collection=` | List documents |
| POST | `/api/documents` | Create a document (JSON: `{name, text, collection?}`) |
| POST | `/api/upload` | Upload a file (multipart) |
| DELETE | `/api/documents/{id}` | Delete a document |
| POST | `/api/ask` | Ask a question (JSON: `{question, top_k?, history?, collection?}`) — one shot |
| POST | `/api/ask/stream` | Same body, **SSE stream**: `trace` / `delta` / `done` events |

SSE events (`data: {...}`):

| Event | Payload | Meaning |
|---|---|---|
| `trace` | `{queries, rewrite, rerank, candidates, citations}` | Retrieval finished; citations and trace pushed first |
| `delta` | `{text}` | Incremental answer text |
| `done` | `{answer, citations, latency_ms, tokens, trace}` | Turn finished |
| `error` | `{detail}` | Failure |

Rate limiting: 60 questions per IP per minute by default (`RAG_RATE_LIMIT`, `0` disables). Over the limit returns `429`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"什么是 RAG？\"}"
```

## Project structure

```
rag-knowledge-base/
├── app/
│   ├── main.py        # FastAPI entry + routes
│   ├── config.py      # Configuration (env / .env)
│   ├── loaders.py     # File → text (pdf/md/txt)
│   ├── chunking.py    # Text chunking (sentences + overlap)
│   ├── embeddings.py  # Embedding model (bge-small-zh / hash fallback)
│   ├── vectorstore.py # Vector index (numpy cosine search)
│   ├── retrieval.py   # BM25 + vector hybrid retrieval (RRF fusion)
│   ├── rewrite.py     # Query rewriting (multi-query)
│   ├── rerank.py      # LLM reranking
│   ├── store.py       # Document store (JSON persistence, default)
│   ├── db.py          # MySQL store + automatic fallback
│   ├── llm.py         # DeepSeek client (retries + errors)
│   └── rag.py         # RAG pipeline (rewrite → hybrid recall → fuse → rerank → generate)
├── static/            # Frontend (no build tools)
├── data/
│   ├── sample/        # Sample knowledge base (incl. distractors for eval)
│   ├── eval/          # Eval set (questions.json) and report (report.json)
│   └── db.json        # Generated at runtime (gitignored; not used with MySQL)
├── scripts/
│   ├── evaluate.py    # Retrieval eval: hit@1 / hit@3 / MRR across configs
│   └── smoke.py       # Real API smoke test
├── docs/
│   └── LEARNING-RAG-ADVANCED.md  # Beginner-friendly deep dive (theory + eval + interview Q&A)
├── tests/             # pytest tests (46 cases)
└── .github/workflows/ci.yml
```

## Running tests

```bash
pytest -q
```

## Retrieval evaluation

The repo ships 100 questions (basics, paraphrases, advanced concepts, section-targeted and distractor questions) and `scripts/evaluate.py`, which runs four configurations and prints retrieval quality side by side:

```bash
python scripts/evaluate.py          # 100 questions; requires a key in .env (rerank/rewrite call the LLM)
python scripts/evaluate.py --no-llm # BM25 vs hybrid only, no API cost
```

Metrics: **hit@1** = share of questions whose gold chunk ranks first; **hit@3** = ranks in the top 3; **MRR** = mean reciprocal rank of the gold chunk.

Baseline (14 chunks, 100 questions, bge-small-zh):

| Configuration | hit@1 | hit@3 | MRR | Latency |
|---|---|---|---|---|
| BM25 only | **0.72** | 0.93 | 0.818 | 3 ms |
| Hybrid recall (BM25 + vectors, RRF) | 0.67 | 0.91 | 0.790 | 14 ms |
| Hybrid + LLM rerank | 0.74 | 0.94 | 0.838 | 1.49 s |
| Hybrid + query rewriting + rerank | **0.75** | **0.96** | **0.848** | 2.64 s |

Breakdown by question type makes it clear who each change helps (hit@1):

| Question group | # | BM25 only | Hybrid | Hybrid + rerank |
|---|---|---|---|---|
| Basics q01–q20 | 20 | 0.70 | 0.65 | **0.80** |
| Paraphrases q21–q28 (colloquial / reworded) | 8 | 0.25 | **0.50** | **0.62** |
| Advanced concepts q29–q62 | 34 | **0.74** | 0.71 | 0.71 |
| Section / distractor q63–q100 | 38 | **0.82** | 0.68 | 0.76 |

> This table is more interesting than the total: **on paraphrase questions BM25 scores only 0.25, hybrid recall doubles it to 0.50 and rerank lifts it to 0.62** — that is the problem vectors + rerank actually solve. Conversely, section questions are largely copied from headings/keywords, where BM25 is close to copying from an open book (0.82) and semantic changes only dilute the lexical edge.
> Hence: never quote a single total. On a small corpus the total hides structural differences, so evaluate per group. The cost is visible too — latency grows from 3 ms to 2.6 s because rerank and rewrite call the LLM; in production, cache the expensive parts (see below).

### Cache

Rewrite and rerank results are cached in SQLite keyed by question + candidate fingerprint (`RAG_CACHE_TTL`, 1 hour by default). Same questions asked again:

| Scenario | Latency |
|---|---|
| First time (cold: one rewrite + one rerank LLM call) | 1.4 – 2.6 s |
| Cache hit (rewrites and rerank scores read from SQLite) | **28 ms** (~50× faster) |

### Rerank backends

| Backend | How it works | Latency | Dependency |
|---|---|---|---|
| `llm` (default) | ask the LLM to score candidates 0–10 | ~1.4 s | API key |
| `cross` | local bge-reranker cross-encoder (fastembed/ONNX) | tens of ms | ~100 MB model download |

If the `cross` model cannot be downloaded or inference fails, the pipeline silently falls back to `llm` reranking.

### Generation quality evaluation (LLM-as-a-judge)

Retrieval metrics only tell you whether the right chunk was found; users see the answer. `scripts/evaluate_generation.py` runs the full RAG pipeline on each question and asks a judge model to score it, RAGAS-style:

```bash
python scripts/evaluate_generation.py --limit 20   # 20 questions, 2 LLM calls each (generate + judge)
```

| Metric | Meaning | Score (20 questions, 0-100) |
|---|---|---|
| Faithfulness | Are the answer's claims supported by the cited source text? | **100** |
| Answer relevance | Does it directly and fully answer the question? | **99** |
| Citation quality | Do the cited chunks actually support the answer? | **97** |
| Refusal | When the corpus really does not cover it, does it say so honestly? | 1 / 20 |

> Two honest caveats:
> 1. The judge receives the **full retrieved chunk text**, not the 160-char snippet shown in the UI — otherwise it mistakes "the snippet does not cover the claim" for "the model hallucinated". The first version of this script fell into exactly that trap and scored 65 / 75 / 63;
> 2. The judge is the **same model** being evaluated, so it is lenient. Absolute numbers only show a trend; this is useful for A/B comparisons (prompt changes, model swaps, reranking on/off). In production, use a stronger judge model plus human spot checks.

### Vector index backends (numpy / FAISS)

| Backend | Suitable size | Notes |
|---|---|---|
| `numpy` (default fallback) | up to a few thousand chunks | one matrix multiply gives exact cosine scores; zero dependencies |
| `faiss` (HNSW) | 10k+ chunks | approximate nearest neighbour, near-logarithmic queries; `M` controls links per node, `efSearch` controls search width — both trade accuracy for speed |

With `RAG_VECTOR_BACKEND=auto`: FAISS HNSW is used when installed; otherwise (or if index creation fails) it silently falls back to numpy. The sample corpus only has 14 chunks, so latency differences are negligible — **the point of FAISS is scale**, and the exercise here is to show *when* to switch indexes, not to shave microseconds.

## Optional: MySQL storage

Switching storage from JSON to MySQL needs **no code changes** — just configuration:

**1. Create database and user (once)**

```sql
CREATE DATABASE rag_kb DEFAULT CHARACTER SET utf8mb4;
CREATE USER 'dsh_app'@'localhost' IDENTIFIED BY 'your-password';
GRANT ALL PRIVILEGES ON rag_kb.* TO 'dsh_app'@'localhost';
```

**2. Fill the database settings in `.env`**

```ini
RAG_DB_HOST=127.0.0.1
RAG_DB_PORT=3306
RAG_DB_USER=dsh_app
RAG_DB_PASSWORD=your-password
RAG_DB_NAME=rag_kb
```

**3. Restart the server.** When `/api/stats` returns `"storage": "mysql"` you are done (tables are created automatically). Open `rag_kb` in DataGrip / Navicat to inspect the data.

**Schema** (see `app/db.py`):

| Table | Columns | Notes |
|---|---|---|
| `documents` | id, name, chunks, chars, created_at | one row per document |
| `chunks` | id, doc_id, idx, text, chars | one row per chunk; `doc_id` FK → `documents` |

**Design points:**

- **Foreign key + cascade delete**: `ON DELETE CASCADE` removes all chunks of a deleted document — no orphan rows.
- **Index**: an index on `doc_id` avoids full scans when listing chunks or cascading deletes.
- **Fallback**: if MySQL is unreachable the app falls back to JSON (`make_store()`), so the demo runs anywhere.
- **Trade-off**: a fresh connection per operation + autocommit is fine for a demo; production should use a pool, and full-text/vector indexes on `text`.

## Design notes

**Why upgrade from BM25 to hybrid retrieval (instead of going all-in on vectors)?**

- BM25 matches terms, costs milliseconds and has zero dependencies — very accurate when the question shares the document's wording;
- Vectors cover paraphrases: "the model keeps answering from old docs after I updated them" vs. "an external knowledge base can be updated anytime" — no lexical overlap, but semantically close;
- Each retriever recalls `RAG_CANDIDATE_K` candidates and the two ranked lists are merged with **RRF (Reciprocal Rank Fusion)**: it only looks at ranks, not scores, which neatly avoids the "BM25 score vs cosine similarity are different scales" problem — `score = Σ 1/(k + rank)`;
- `bge-small-zh-v1.5` is 512-dimensional, runs on CPU and needs no vector database; following bge's recommendation the query side gets an instruction prefix while the document side does not.

**Why LLM reranking and query rewriting?**

- Reranking: recall favors "don't miss" (top-12), rerank favors "rank precisely" (the LLM scores each candidate, take top-4);
- Rewriting: user questions are colloquial and sparse; asking the LLM for 2–3 retrieval queries and fusing them lifts hit rate (paraphrase questions 0.50 → 0.62, overall 0.74 → 0.75);
- Both cost latency (milliseconds → seconds), so both are switches: measurable and easy to turn off.

**Why grow the eval set to 100 questions — and split it by type?**

- Going from 62 to 100 questions changes the mix (more concept and distractor questions), so absolute scores shift — the exam changed, not the model. Always report the size and composition;
- A single total can make "BM25 beats hybrid" look true (0.72 vs 0.67), because many questions reuse the document's exact terms — lexical matching is nearly cheating there;
- Split by group and the story gets honest: **paraphrases 0.25 → 0.50 (hybrid) → 0.62 (rerank)**, while section questions stay at 0.82 for BM25. In an interview, "how I noticed the eval set was biased and how I attributed gains per group" beats reciting a strong total.

**Why add a FAISS vector index?**

- numpy exact search is O(N·D): beyond tens of thousands of chunks every query scans the whole corpus and latency grows linearly;
- FAISS HNSW builds a multi-layer graph for approximate nearest neighbours, near-logarithmic queries, at the cost of recall < 100% (tune `efSearch` to trade accuracy);
- Shipped as an `auto` backend with graceful fallback to numpy: zero dependencies on small corpora, and a demonstrable answer to "what do you switch when the corpus grows" — the same retrieval interface with two implementations is the "evolvable architecture" this kind of project should show.

**Why collections?**

- Dumping every document into one knowledge base pollutes retrieval: ask about "reimbursement" and a travel guide may still show up as a candidate chunk;
- Each document carries a `collection` field (default `default`) and retrieval only runs inside that collection — data isolation on top of hybrid retrieval;
- MySQL filters on an indexed `collection` column; the JSON backend filters in Python; the sidebar switches/creates collections and the document list, stats and Q&A all follow;
- Be honest in an interview: this is **logical isolation**, not an auth system — real multi-tenant work still needs identity, per-collection permissions and quotas.

**Why streaming, multi-turn and rate limiting?**

- Streaming: retrieval + generation takes 1–2 s, and the page sits blank until it finishes. Over SSE the retrieval trace and citations arrive first, then the answer streams token by token;
- Multi-turn: follow-ups ("and its downsides?") would be treated as brand-new questions without context; the frontend sends the last few turns so it behaves like a chat assistant;
- Rate limiting: answers cost tokens, and a public demo can be abused. Cap each IP per minute (`RAG_RATE_LIMIT`), return 429, and let health checks through.

**Fallback strategy (a common interview question)**

- Embedding model fails to load → degrade to the hash embedder (`HashEmbedder`) so the service keeps running;
- One retriever errors → degrade to pure BM25;
- Rerank/rewrite output is not valid JSON → keep the original order (rerank) or fall back to the original question (rewrite);
- MySQL unreachable → degrade to JSON storage.

Correctness is covered by tests: `tests/test_hybrid.py` and `tests/test_pipeline.py` exercise fusion ordering, bad-output fallbacks and the trace fields.

**How to make it production-ready?**

1. Storage: JSON (default) → MySQL (supported) → vector DB (Chroma / Milvus);
2. Retrieval: hybrid retrieval (supported) → HNSW index in a vector DB + multi-recall;
3. Chunking: structure-aware splitting (Markdown headings, PDF layout);
4. Evaluation: already 100 questions with per-group attribution and an LLM-judge generation eval; next steps are official RAGAS metrics and a human spot-check set;
5. Engineering: access control, logging & monitoring, rerank caching (done), multi-turn context (done), rate limiting (done).

## Roadmap

- [x] MySQL storage backend (documents / chunks tables)
- [x] Vector retrieval (bge-small-zh) + BM25 hybrid retrieval (RRF fusion)
- [x] LLM reranking + query rewriting
- [x] Retrieval evaluation (62 questions, hit@1 / hit@3 / MRR, multi-config comparison)
- [x] Streaming answers (SSE: retrieval trace + token deltas)
- [x] Multi-turn conversation context (frontend sends recent turns)
- [x] Per-IP rate limiting (`RAG_RATE_LIMIT`)
- [x] Vector index backends: numpy exact / FAISS HNSW approximate (graceful fallback)
- [x] Eval set grown to 100 questions with per-group attribution
- [x] Generation quality evaluation (faithfulness / relevance / citation, LLM-as-a-judge)
- [x] Collections (isolated knowledge bases, switchable in the UI)
- [ ] Answer rating / feedback button
- [ ] Docker deployment

## Security notes

`.env` (API key and database password) and `data/db.json` are ignored by `.gitignore` and never committed; this README and the sample config contain no real secrets.

## FAQ

- **Error `没有找到 DEEPSEEK_API_KEY`**: check that `.env` is in the project root and has a value, or set the environment variable and restart the terminal.
- **404 / model not found**: set `DEEPSEEK_MODEL` back to `deepseek-chat` in `.env`.
- **Answer "知识库里没有找到"**: make sure the sidebar lists documents; if the wording differs a lot from the source, the hybrid retriever can usually still find it — if not, try rephrasing or a different keyword.
- **PDF has no extractable text**: scanned PDFs are images and need OCR (not supported yet).

## License

[MIT](LICENSE)

