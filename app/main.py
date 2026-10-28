"""FastAPI 入口：上传文档 / 提问 / 查看知识库，并托管前端页面。

启动方式（项目根目录下）：
    uvicorn app.main:app --reload
然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .config import DATA_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, RATE_LIMIT_PER_MIN, STATIC_DIR
from .db import make_store
from .loaders import load_bytes
from .rag import answer as rag_answer
from .rag import answer_stream as rag_answer_stream

store = make_store()


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


class DocumentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": DEEPSEEK_MODEL, "has_key": bool(DEEPSEEK_API_KEY)}


@app.get("/api/stats")
def stats() -> dict:
    return {
        **store.stats(),
        "has_key": bool(DEEPSEEK_API_KEY),
        "model": DEEPSEEK_MODEL,
        "storage": store.backend,
    }


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return store.list_documents()


@app.post("/api/documents", status_code=201)
def create_document(payload: DocumentIn) -> dict:
    try:
        doc = store.add(payload.name, payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": doc.id, "name": doc.name, "chunks": len(doc.chunks)}


@app.post("/api/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)) -> dict:
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
        doc = store.add(file.filename or "upload.txt", text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": doc.id, "name": doc.name, "chunks": len(doc.chunks)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not store.delete(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"ok": True}


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict:
    history = [m.model_dump() for m in payload.history]
    try:
        return rag_answer(payload.question, store.all_chunks(), payload.top_k, history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):
    """流式回答：SSE，每行一个 JSON 事件（trace / delta / done / error）。"""
    history = [m.model_dump() for m in payload.history]
    chunks = store.all_chunks()

    def event_stream():
        try:
            for event in rag_answer_stream(payload.question, chunks, payload.top_k, history):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
        except llm.LLMError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
