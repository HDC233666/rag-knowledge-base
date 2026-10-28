"""文档仓库：文档 + 切分块，持久化到 data/db.json（重启不丢）。

v5 起支持「多知识库」：每篇文档属于一个 collection（默认 default），
检索时只取当前知识库的切片，互不干扰。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import split_text
from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR

DEFAULT_DB = DATA_DIR / "db.json"
DEFAULT_COLLECTION = "default"


@dataclass
class Document:
    id: str
    name: str
    text: str
    created_at: float
    collection: str = DEFAULT_COLLECTION
    chunks: list[dict] = field(default_factory=list)


class DocumentStore:
    """一个极简的文档仓库（JSON 存储，天然适合 demo 和小规模使用）。"""

    backend = "json"

    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.documents: dict[str, Document] = {}
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if not self.db_path.exists():
            return
        raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        for item in raw.get("documents", []):
            item.setdefault("collection", DEFAULT_COLLECTION)  # 兼容 v4 及更早的 db.json
            self.documents[item["id"]] = Document(**item)

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": [asdict(doc) for doc in self.documents.values()]}
        self.db_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ---------- 增删查 ----------
    def add(self, name: str, text: str, collection: str = DEFAULT_COLLECTION) -> Document:
        text = (text or "").strip()
        if not text:
            raise ValueError("文档内容为空")
        doc = Document(
            id=uuid.uuid4().hex[:12],
            name=name,
            text=text,
            created_at=time.time(),
            collection=collection or DEFAULT_COLLECTION,
        )
        pieces = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        doc.chunks = [
            {
                "id": f"{doc.id}:{index}",
                "doc_id": doc.id,
                "doc_name": name,
                "collection": doc.collection,
                "index": index,
                "text": piece,
            }
            for index, piece in enumerate(pieces)
        ]
        self.documents[doc.id] = doc
        self._save()
        return doc

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self._save()
        return True

    def get(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def list_documents(self, collection: str | None = None) -> list[dict]:
        docs = sorted(self.documents.values(), key=lambda d: d.created_at, reverse=True)
        if collection is not None:
            docs = [doc for doc in docs if doc.collection == collection]
        return [
            {
                "id": doc.id,
                "name": doc.name,
                "collection": doc.collection,
                "created_at": doc.created_at,
                "chunks": len(doc.chunks),
                "chars": len(doc.text),
            }
            for doc in docs
        ]

    def all_chunks(self, collection: str | None = None) -> list[dict]:
        chunks = [
            {"collection": doc.collection, **chunk}
            for doc in self.documents.values()
            for chunk in doc.chunks
        ]
        if collection is not None:
            chunks = [c for c in chunks if c["collection"] == collection]
        return chunks

    def collections(self) -> list[dict]:
        """知识库列表（名称 + 文档数 + 切片数），没有文档时也返回 default。"""
        names: dict[str, dict] = {}
        for doc in self.documents.values():
            item = names.setdefault(
                doc.collection, {"name": doc.collection, "documents": 0, "chunks": 0}
            )
            item["documents"] += 1
            item["chunks"] += len(doc.chunks)
        if DEFAULT_COLLECTION not in names:
            names[DEFAULT_COLLECTION] = {"name": DEFAULT_COLLECTION, "documents": 0, "chunks": 0}
        return sorted(names.values(), key=lambda item: (item["name"] != DEFAULT_COLLECTION, item["name"]))

    def stats(self, collection: str | None = None) -> dict:
        docs = list(self.documents.values())
        if collection is not None:
            docs = [doc for doc in docs if doc.collection == collection]
        return {
            "documents": len(docs),
            "chunks": sum(len(doc.chunks) for doc in docs),
            "chars": sum(len(doc.text) for doc in docs),
        }
