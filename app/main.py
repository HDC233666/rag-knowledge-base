"""FastAPI 入口：上传文档 / 提问 / 查看知识库，并托管前端页面。

启动方式（项目根目录下）：
    uvicorn app.main:app --reload
然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm, metrics
from .config import DATA_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, RATE_LIMIT_PER_MIN, STATIC_DIR
from .db import make_store
from .loaders import load_bytes
from .rag import answer as rag_answer
from .rag import retrieve as rag_retrieve
from .rag import answer_stream as rag_answer_stream
from .vectorstore import backend_name as vector_backend_name

store = make_store()

# 知识库名称：中文、字母、数字、下划线、短横线，1~64 字符
_COLLECTION_RE = re.compile(r"^[\w\u4e00-\u9fff-]{1,64}$")


def _clean_collection(value: str | None) -> str:
    name = (value or "default").strip() or "default"
    if not _COLLECTION_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="知识库名称只能包含中文、字母、数字、下划线和短横线（1~64 字符）",
        )
    return name


def _seed_sample_docs() -> None:
    """启动时同步 data/sample 里的示例文档：缺的补进来，已有的跳过。

    这样升级知识库（新增示例文档）后重启即可自动导入，不用手动清库。
    """
    sample_dir = DATA_DIR / "sample"
    if not sample_dir.exists():
        return
    existing = {doc["name"] for doc in store.list_documents()}
    for path in sorted(sample_dir.glob("*.md")):
        if path.name in existing:
            continue
        try:
            store.add(path.name, path.read_text(encoding="utf-8"))
        except ValueError:
            continue


@asynccontextmanager
async def lifespan(_: FastAPI):
    _seed_sample_docs()
    yield


app = FastAPI(title="RAG 知识库问答", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 限流：每个 IP 每分钟最多 RATE_LIMIT_PER_MIN 个 /api 请求（health 除外），进程内存储
_rate_buckets: dict[str, deque] = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    path = request.url.path
    if RATE_LIMIT_PER_MIN > 0 and path.startswith("/api/") and not path.startswith("/api/health"):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = _rate_buckets.setdefault(ip, deque())
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MIN:
            return JSONResponse(
                {"detail": f"请求太频繁了，请稍后再试（每 IP 每分钟 {RATE_LIMIT_PER_MIN} 次）"},
                status_code=429,
            )
        bucket.append(now)
    return await call_next(request)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    history: list[Message] = Field(default_factory=list, max_length=12)
    collection: str = Field(default="default", max_length=64)


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    collection: str = Field(default="default", max_length=64)
    top_k: int = Field(default=4, ge=1, le=20)
    llm: bool = Field(default=False, description="是否启用查询改写 / LLM 重排（默认关闭，压测与评估用）")


class DocumentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    collection: str = Field(default="default", max_length=64)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": DEEPSEEK_MODEL,
        "has_key": bool(DEEPSEEK_API_KEY),
        "vector_backend": vector_backend_name(),
    }


@app.get("/api/collections")
def list_collections() -> list[dict]:
    """知识库列表（名称 + 文档数 + 切片数），前端用于切换。"""
    return store.collections()


@app.get("/api/stats")
def stats(collection: str | None = None) -> dict:
    name = _clean_collection(collection) if collection else None
    return {
        **store.stats(name),
        "collection": name,
        "has_key": bool(DEEPSEEK_API_KEY),
        "model": DEEPSEEK_MODEL,
        "storage": store.backend,
    }


@app.get("/api/documents")
def list_documents(collection: str | None = None) -> list[dict]:
    name = _clean_collection(collection) if collection else None
    return store.list_documents(name)


@app.post("/api/documents", status_code=201)
def create_document(payload: DocumentIn) -> dict:
    collection = _clean_collection(payload.collection)
    try:
        doc = store.add(payload.name, payload.text, collection)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": doc.id, "name": doc.name, "collection": doc.collection, "chunks": len(doc.chunks)}


@app.post("/api/upload", status_code=201)
async def upload_document(file: UploadFile = File(...), collection: str = Form("default")) -> dict:
    name = _clean_collection(collection)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件是空的")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件太大（上限 20MB）")
    try:
        text = load_bytes(data, file.filename or "upload.txt")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        doc = store.add(file.filename or "upload.txt", text, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    metrics.incr("rag_upload_total")
    return {"id": doc.id, "name": doc.name, "collection": doc.collection, "chunks": len(doc.chunks)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not store.delete(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"ok": True}


@app.post("/api/retrieve")
def retrieve_only(payload: RetrieveRequest) -> dict:
    """检索调试接口：只跑检索链路，不生成回答（评估 / 压测用，默认不花 API 费用）。"""
    name = _clean_collection(payload.collection)
    chunks = store.all_chunks(name)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"知识库「{name}」还没有文档，请先上传。")
    started = time.perf_counter()
    try:
        hits, trace = rag_retrieve(payload.question, chunks, payload.top_k, use_llm=payload.llm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    return {
        "collection": name,
        "hits": hits,
        "trace": trace,
        "latency_ms": latency_ms,
    }


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict:
    history = [m.model_dump() for m in payload.history]
    collection = _clean_collection(payload.collection)
    metrics.incr("rag_ask_total")
    try:
        result = rag_answer(payload.question, store.all_chunks(collection), payload.top_k, history)
    except ValueError as exc:
        metrics.incr("rag_ask_error")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except llm.LLMError as exc:
        metrics.incr("rag_ask_error")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    metrics.incr("rag_ask_ok")
    if result.get("latency_ms") is not None:
        metrics.incr("rag_ask_latency_ms_sum", int(result["latency_ms"]))
    return result


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):
    """流式回答：SSE，每行一个 JSON 事件（trace / delta / done / error）。"""
    history = [m.model_dump() for m in payload.history]
    collection = _clean_collection(payload.collection)
    chunks = store.all_chunks(collection)

    metrics.incr("rag_stream_total")

    def event_stream():
        try:
            for event in rag_answer_stream(payload.question, chunks, payload.top_k, history):
                if event.get("type") == "done":
                    metrics.incr("rag_stream_ok")
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            metrics.incr("rag_stream_error")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
        except llm.LLMError as exc:
            metrics.incr("rag_stream_error")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    question: str = Field("", max_length=2000)
    comment: str = Field("", max_length=2000)
    collection: str = Field("default", max_length=64)


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackRequest) -> dict:
    """记录一次答案反馈（👍/👎），追加写入 data/feedback.jsonl。"""
    record = {
        "rating": payload.rating,
        "question": payload.question.strip(),
        "comment": payload.comment.strip(),
        "collection": _clean_collection(payload.collection),
        "created_at": time.time(),
    }
    path = DATA_DIR / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics.incr(f"rag_feedback_{payload.rating}")
    return {"ok": True}


@app.get("/api/feedback/stats")
def feedback_stats() -> dict:
    path = DATA_DIR / "feedback.jsonl"
    up = down = 0
    recent: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("rating") == "up":
                up += 1
            elif item.get("rating") == "down":
                down += 1
            recent.append(item)
    return {"up": up, "down": down, "total": up + down, "recent": recent[-5:]}


@app.get("/api/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    """Prometheus 文本格式指标：便于接监控，也更像生产服务。"""
    metrics.set_gauge("rag_collections", len(store.collections()))
    return metrics.render_prometheus()
