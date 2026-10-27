"""文档仓库：文档 + 切分块，持久化到 data/db.json（重启不丢）。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import split_text
from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR

DEFAULT_DB = DATA_DIR / "db.json"


@dataclass
class Document:
    id: str
    name: str
    text: str
    created_at: float
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
            self.documents[item["id"]] = Document(**item)

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": [asdict(doc) for doc in self.documents.values()]}
        self.db_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ---------- 增删查 ----------
    def add(self, name: str, text: str) -> Document:
        text = (text or "").strip()
        if not text:
            raise ValueError("文档内容为空")
        doc = Document(
            id=uuid.uuid4().hex[:12],
            name=name,
            text=text,
            created_at=time.time(),
        )
        pieces = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        doc.chunks = [
            {
                "id": f"{doc.id}:{index}",
                "doc_id": doc.id,
                "doc_name": name,
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

    def list_documents(self) -> list[dict]:
        docs = sorted(self.documents.values(), key=lambda d: d.created_at, reverse=True)
        return [
            {
                "id": doc.id,
                "name": doc.name,
                "created_at": doc.created_at,
                "chunks": len(doc.chunks),
                "chars": len(doc.text),
            }
            for doc in docs
        ]

    def all_chunks(self) -> list[dict]:
        return [chunk for doc in self.documents.values() for chunk in doc.chunks]

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.all_chunks()),
            "chars": sum(len(doc.text) for doc in self.documents.values()),
        }
