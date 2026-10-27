"""MySQL 存储后端（可选增强）。

配置了 ``RAG_DB_HOST`` 时启用；没配置或连不上时**自动降级**到 JSON 存储，
保证项目在任何电脑上都能直接跑起来（面试时这也是一个可以讲的设计点）。

表结构：

    documents(id, name, chunks, chars, created_at)
    chunks(id, doc_id, idx, text, chars)      -- doc_id 外键，删文档时级联删块

索引设计：
- ``pk_documents``：主键自增，查询快
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
from .store import Document, DocumentStore

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        name       VARCHAR(255)    NOT NULL,
        chunks     INT             NOT NULL DEFAULT 0,
        chars      INT             NOT NULL DEFAULT 0,
        created_at TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id)
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

    # ---------- 增删查 ----------
    def add(self, name: str, text: str) -> Document:
        text = (text or "").strip()
        if not text:
            raise ValueError("文档内容为空")
        pieces = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (name, chunks, chars) VALUES (%s, %s, %s)",
                    (name, len(pieces), len(text)),
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
            chunks=[
                {
                    "id": f"{doc_id}:{index}",
                    "doc_id": str(doc_id),
                    "doc_name": name,
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

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, created_at, chunks, chars "
                    "FROM documents ORDER BY created_at DESC, id DESC"
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "created_at": row["created_at"].timestamp(),
                "chunks": int(row["chunks"]),
                "chars": int(row["chars"]),
            }
            for row in rows
        ]

    def all_chunks(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT c.doc_id, c.idx, c.text, d.name AS doc_name "
                    "FROM chunks c JOIN documents d ON d.id = c.doc_id "
                    "ORDER BY c.doc_id, c.idx"
                )
                rows = cur.fetchall()
        return [
            {
                "id": f"{row['doc_id']}:{row['idx']}",
                "doc_id": str(row["doc_id"]),
                "doc_name": row["doc_name"],
                "index": int(row["idx"]),
                "text": row["text"],
            }
            for row in rows
        ]

    def stats(self) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS documents, "
                    "COALESCE(SUM(chunks), 0) AS chunks, "
                    "COALESCE(SUM(chars), 0) AS chars FROM documents"
                )
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
