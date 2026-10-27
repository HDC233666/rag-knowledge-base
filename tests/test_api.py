from fastapi.testclient import TestClient

import app.main as main
from app.store import DocumentStore

client = TestClient(main.app)


def _use_temp_store(tmp_path, monkeypatch):
    store = DocumentStore(tmp_path / "db.json")
    monkeypatch.setattr(main, "store", store)
    return store


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_upload_and_ask(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.rag.llm.chat",
        lambda messages, **kwargs: {"text": "RAG 是检索增强生成 [1]", "latency_ms": 5, "tokens": 30},
    )

    upload = client.post(
        "/api/upload",
        files={
            "file": (
                "note.md",
                "RAG 是检索增强生成。\n先检索资料，再让模型回答。".encode("utf-8"),
                "text/markdown",
            )
        },
    )
    assert upload.status_code == 201
    assert upload.json()["chunks"] >= 1

    answer = client.post("/api/ask", json={"question": "什么是 RAG？"})
    assert answer.status_code == 200
    body = answer.json()
    assert "RAG" in body["answer"]
    assert body["citations"]


def test_upload_rejects_unsupported_type(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    response = client.post(
        "/api/upload",
        files={"file": ("bad.exe", b"MZ\x00\x01", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_delete_missing_document_returns_404(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    assert client.delete("/api/documents/not-exist").status_code == 404


def test_ask_without_chunks_returns_friendly_message(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    answer = client.post("/api/ask", json={"question": "什么是向量检索？"})
    assert answer.status_code == 200
    assert "没有找到" in answer.json()["answer"]
