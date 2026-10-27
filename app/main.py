"""FastAPI 入口：上传文档 / 提问 / 查看知识库，并托管前端页面。

启动方式（项目根目录下）：
    uvicorn app.main:app --reload
然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .config import DATA_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, STATIC_DIR
from .db import make_store
from .loaders import load_bytes
from .rag import answer as rag_answer

store = make_store()


def _seed_sample_docs() -> None:
    """首次启动时把 data/sample 里的示例文档导入，方便直接体验。"""
    if store.stats()["documents"] > 0:
        return
    sample_dir = DATA_DIR / "sample"
    if not sample_dir.exists():
        return
    for path in sorted(sample_dir.glob("*.md")):
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


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


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
    try:
        return rag_answer(payload.question, store.all_chunks(), payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
