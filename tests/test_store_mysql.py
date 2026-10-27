"""MySQL 存储后端的集成测试。

没有配置 MySQL（RAG_DB_HOST / RAG_DB_NAME）时自动跳过，所以 CI 上不用装数据库。
本地想跑：填好 .env 里的 RAG_DB_* 后执行 `python -m pytest tests/test_store_mysql.py -v`
"""

from __future__ import annotations

import pytest

from app.config import DB_HOST, DB_NAME
from app.db import MySQLDocumentStore

pytestmark = pytest.mark.skipif(
    not (DB_HOST and DB_NAME),
    reason="需要 MySQL 环境（设置 RAG_DB_HOST / RAG_DB_NAME）",
)


def test_mysql_store_crud():
    store = MySQLDocumentStore()
    name = "集成测试文档-mysql"
    text = "MySQL 是关系型数据库。" * 120  # 够长，会切出多个块

    doc = store.add(name, text)
    try:
        docs = store.list_documents()
        assert any(d["name"] == name for d in docs)

        chunks = store.all_chunks()
        mine = [c for c in chunks if c["doc_name"] == name]
        assert mine, "应能在 MySQL 里查到刚写入的块"
        assert all(c["text"].strip() for c in mine)

        stats = store.stats()
        assert stats["documents"] >= 1 and stats["chunks"] >= 1

        assert store.delete(doc.id) is True
        assert not any(d["name"] == name for d in store.list_documents())
        assert store.delete(doc.id) is False  # 再删一次应返回 False
    finally:  # 兜底清理，避免测试污染演示数据
        for item in store.list_documents():
            if item["name"] == name:
                store.delete(item["id"])
