"""多知识库（collection）测试：存储层隔离 + API 过滤 + 迁移兼容。"""

import json

from fastapi.testclient import TestClient

import app.main as main
from app.store import DocumentStore

client = TestClient(main.app)


def _use_temp_store(tmp_path, monkeypatch):
    store = DocumentStore(tmp_path / "db.json")
    monkeypatch.setattr(main, "store", store)
    return store


# ---------- 存储层 ----------


def test_store_isolates_collections(tmp_path):
    store = DocumentStore(tmp_path / "db.json")
    store.add("a.md", "RAG 是检索增强生成。", collection="alpha")
    store.add("b.md", "BM25 是稀疏检索。", collection="beta")
    store.add("c.md", "默认知识库的文档。")

    assert {doc["name"] for doc in store.list_documents("alpha")} == {"a.md"}
    assert {doc["name"] for doc in store.list_documents("beta")} == {"b.md"}
    assert {doc["name"] for doc in store.list_documents("default")} == {"c.md"}
    assert len(store.list_documents()) == 3  # 不传 = 全部

    assert all(chunk["collection"] == "alpha" for chunk in store.all_chunks("alpha"))
    assert store.stats("alpha")["documents"] == 1
    assert store.stats()["documents"] == 3


def test_store_collections_summary(tmp_path):
    store = DocumentStore(tmp_path / "db.json")
    store.add("a.md", "第一段。\n\n第二段。", collection="alpha")
    store.add("b.md", "内容。", collection="beta")

    summary = {item["name"]: item for item in store.collections()}
    assert summary["alpha"]["documents"] == 1
    assert summary["alpha"]["chunks"] >= 1
    assert summary["beta"]["documents"] == 1
    assert "default" in summary and summary["default"]["documents"] == 0


def test_store_loads_legacy_db_without_collection(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "abc123",
                        "name": "old.md",
                        "text": "旧版本文档",
                        "created_at": 1.0,
                        "chunks": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = DocumentStore(path)
    assert store.documents["abc123"].collection == "default"


# ---------- API 层 ----------


def test_api_upload_into_collection(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    response = client.post(
        "/api/upload",
        files={"file": ("work.md", "工作知识库的内容。".encode("utf-8"), "text/markdown")},
        data={"collection": "work"},
    )
    assert response.status_code == 201
    assert response.json()["collection"] == "work"

    docs = client.get("/api/documents", params={"collection": "work"}).json()
    assert [doc["name"] for doc in docs] == ["work.md"]
    assert client.get("/api/documents", params={"collection": "default"}).json() == []


def test_api_collection_switch_is_isolation(tmp_path, monkeypatch):
    """A 知识库的文档不能出现在 B 知识库的检索结果里。"""
    _use_temp_store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.rag.llm.chat",
        lambda messages, **kwargs: {"text": "苹果是水果 [1]", "latency_ms": 5, "tokens": 20},
    )
    client.post(
        "/api/upload",
        files={"file": ("alpha.md", "苹果是水果。".encode("utf-8"), "text/markdown")},
        data={"collection": "alpha"},
    )
    client.post(
        "/api/upload",
        files={"file": ("beta.md", "香蕉是水果。".encode("utf-8"), "text/markdown")},
        data={"collection": "beta"},
    )

    answer = client.post(
        "/api/ask", json={"question": "苹果是什么？", "collection": "alpha"}
    ).json()
    names = {citation["doc_name"] for citation in answer["citations"]}
    assert "alpha.md" in names
    assert "beta.md" not in names

    empty = client.post(
        "/api/ask", json={"question": "苹果是什么？", "collection": "empty"}
    ).json()
    assert empty["citations"] == []
    assert "没有找到" in empty["answer"]


def test_api_stats_and_collections(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    client.post(
        "/api/upload",
        files={"file": ("x.md", "内容。".encode("utf-8"), "text/markdown")},
        data={"collection": "work"},
    )
    stats = client.get("/api/stats", params={"collection": "work"}).json()
    assert stats["documents"] == 1
    assert stats["collection"] == "work"

    collections = {item["name"]: item for item in client.get("/api/collections").json()}
    assert collections["work"]["documents"] == 1
    assert "default" in collections


def test_api_rejects_invalid_collection_name(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    assert client.get("/api/stats", params={"collection": "bad name!"}).status_code == 400
    response = client.post(
        "/api/upload",
        files={"file": ("a.md", "x".encode("utf-8"), "text/markdown")},
        data={"collection": "../../etc"},
    )
    assert response.status_code == 400
