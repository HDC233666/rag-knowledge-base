"""MySQL 存储后端（可选增强）。

配置了 ``RAG_DB_HOST`` 时启用；没配置或连不上时**自动降级**到 JSON 存储，
保证项目在任何电脑上都能直接跑起来（面试时这也是一个可以讲的设计点）。

表结构（v5 起支持多知识库）：

    documents(id, name, collection, chunks, chars, created_at)
    chunks(id, doc_id, idx, text, chars)      -- doc_id 外键，删文档时级联删块

索引设计：
- ``pk_documents``：主键自增，查询快
- ``idx_documents_collection``：按知识库过滤文档（多知识库的核心查询）
- ``idx_chunks_doc``：按 doc_id 找某个文档的所有块（列表/删除/级联都靠它）

连接策略：演示项目用「每次操作建连接 + autocommit」的简单写法，
生产环境应换成连接池（如 SQLAlchemy pool / DBUtils PooledDB）。
"""

from __future__ import annotations

import time

import pymysql
from pymysql.cursors import DictCursor

from .chunking import split_text
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
)
from .store import DEFAULT_COLLECTION, Document, DocumentStore

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        name       VARCHAR(255)    NOT NULL,
        collection VARCHAR(64)     NOT NULL DEFAULT 'default',
        chunks     INT             NOT NULL DEFAULT 0,
        chars      INT             NOT NULL DEFAULT 0,
        created_at TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY idx_documents_collection (collection)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        doc_id BIGINT UNSIGNED NOT NULL,
        idx    INT             NOT NULL,
        text   MEDIUMTEXT      NOT NULL,
        chars  INT             NOT NULL DEFAULT 0,
        PRIMARY KEY (id),
        KEY idx_chunks_doc (doc_id),
        CONSTRAINT fk_chunks_doc FOREIGN KEY (doc_id)
            REFERENCES documents (id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


class MySQLDocumentStore:
    """与 JSON DocumentStore 同接口的 MySQL 实现。"""

    backend = "mysql"

    def __init__(self) -> None:
        self._init_schema()

    # ---------- 连接与建表 ----------
    def _connect(self) -> pymysql.connections.Connection:
        return pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
            connect_timeout=5,
        )

    def _init_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                for sql in SCHEMA_STATEMENTS:
                    cur.execute(sql)
                # v4 → v5 迁移：老库没有 collection 列时补上（幂等）
                cur.execute(
                    "SELECT COUNT(*) AS n FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'documents' "
                    "AND COLUMN_NAME = 'collection'",
                    (DB_NAME,),
                )
                if int((cur.fetchone() or {}).get("n") or 0) == 0:
                    cur.execute(
                        "ALTER TABLE documents ADD COLUMN collection VARCHAR(64) "
                        f"NOT NULL DEFAULT '{DEFAULT_COLLECTION}' AFTER name"
                    )
                    cur.execute(
                        "CREATE INDEX idx_documents_collection ON documents (collection)"
                    )

    # ---------- 增删查 ----------
    def add(self, name: str, text: str, collection: str = DEFAULT_COLLECTION) -> Document:
        text = (text or "").strip()
        if not text:
            raise ValueError("文档内容为空")
        collection = collection or DEFAULT_COLLECTION
        pieces = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (name, collection, chunks, chars) VALUES (%s, %s, %s, %s)",
                    (name, collection, len(pieces), len(text)),
                )
                doc_id = int(cur.lastrowid or 0)
                cur.executemany(
                    "INSERT INTO chunks (doc_id, idx, text, chars) VALUES (%s, %s, %s, %s)",
                    [(doc_id, index, piece, len(piece)) for index, piece in enumerate(pieces)],
                )
        return Document(
            id=str(doc_id),
            name=name,
            text=text,
            created_at=time.time(),
            collection=collection,
            chunks=[
                {
                    "id": f"{doc_id}:{index}",
                    "doc_id": str(doc_id),
                    "doc_name": name,
                    "collection": collection,
                    "index": index,
                    "text": piece,
                }
                for index, piece in enumerate(pieces)
            ],
        )

    def delete(self, doc_id: str) -> bool:
        try:
            key = int(doc_id)
        except (TypeError, ValueError):
            return False
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (key,))
                return cur.rowcount > 0

    def list_documents(self, collection: str | None = None) -> list[dict]:
        sql = (
            "SELECT id, name, collection, created_at, chunks, chars FROM documents "
        )
        params: tuple = ()
        if collection is not None:
            sql += "WHERE collection = %s "
            params = (collection,)
        sql += "ORDER BY created_at DESC, id DESC"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "collection": row["collection"],
                "created_at": row["created_at"].timestamp(),
                "chunks": int(row["chunks"]),
                "chars": int(row["chars"]),
            }
            for row in rows
        ]

    def all_chunks(self, collection: str | None = None) -> list[dict]:
        sql = (
            "SELECT c.doc_id, c.idx, c.text, d.name AS doc_name, d.collection "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id "
        )
        params: tuple = ()
        if collection is not None:
            sql += "WHERE d.collection = %s "
            params = (collection,)
        sql += "ORDER BY c.doc_id, c.idx"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            {
                "id": f"{row['doc_id']}:{row['idx']}",
                "doc_id": str(row["doc_id"]),
                "doc_name": row["doc_name"],
                "collection": row["collection"],
                "index": int(row["idx"]),
                "text": row["text"],
            }
            for row in rows
        ]

    def collections(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT collection AS name, COUNT(*) AS documents, "
                    "COALESCE(SUM(chunks), 0) AS chunks "
                    "FROM documents GROUP BY collection ORDER BY collection"
                )
                rows = cur.fetchall()
        result = [
            {"name": row["name"], "documents": int(row["documents"]), "chunks": int(row["chunks"])}
            for row in rows
        ]
        if DEFAULT_COLLECTION not in {item["name"] for item in result}:
            result.insert(0, {"name": DEFAULT_COLLECTION, "documents": 0, "chunks": 0})
        return result

    def stats(self, collection: str | None = None) -> dict:
        sql = "SELECT COUNT(*) AS documents, COALESCE(SUM(chunks), 0) AS chunks, COALESCE(SUM(chars), 0) AS chars FROM documents"
        params: tuple = ()
        if collection is not None:
            sql += " WHERE collection = %s"
            params = (collection,)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone() or {}
        return {
            "documents": int(row.get("documents") or 0),
            "chunks": int(row.get("chunks") or 0),
            "chars": int(row.get("chars") or 0),
        }


def make_store() -> DocumentStore | MySQLDocumentStore:
    """根据配置选择存储后端；MySQL 不可用时降级 JSON，保证服务能启动。"""
    if DB_HOST and DB_NAME and DB_USER:
        try:
            store = MySQLDocumentStore()
            print(f"[store] 已连接 MySQL {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
            return store
        except Exception as exc:  # noqa: BLE001 —— 演示项目：数据库挂了也别让整个服务起不来
            print(f"[store] MySQL 连接失败，自动降级为 JSON 存储：{exc}")
    return DocumentStore()
