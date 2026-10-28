import json

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


def _parse_sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_stream_answer(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)

    def fake_stream(messages, **kwargs):
        yield {"delta": "RAG 是检索增强生成"}
        yield {"delta": " [1]"}
        yield {"done": True, "tokens": 12, "latency_ms": 3}

    monkeypatch.setattr("app.rag.llm.chat_stream", fake_stream)

    upload = client.post(
        "/api/upload",
        files={"file": ("note.md", "RAG 是检索增强生成。先检索再回答。".encode("utf-8"), "text/markdown")},
    )
    assert upload.status_code == 201

    response = client.post("/api/ask/stream", json={"question": "什么是 RAG？"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    types = [e["type"] for e in events]
    # 第一个事件可能是 stage（检索前提示），trace 必须出现
    assert "trace" in types
    assert types.index("trace") < len(types) - 1
    assert events[-1]["type"] == "done"
    delta_text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "RAG" in delta_text
    assert "RAG" in events[-1]["answer"]
    assert events[-1]["citations"]
    assert events[-1]["tokens"] == 12


def test_stream_accepts_history(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    captured = {}

    def fake_stream(messages, **kwargs):
        captured["messages"] = messages
        yield {"delta": "好的"}
        yield {"done": True, "tokens": 1, "latency_ms": 1}

    monkeypatch.setattr("app.rag.llm.chat_stream", fake_stream)
    client.post("/api/upload", files={"file": ("a.md", "RAG 简介".encode("utf-8"), "text/markdown")})

    response = client.post(
        "/api/ask/stream",
        json={
            "question": "它有什么好处？",
            "history": [
                {"role": "user", "content": "什么是 RAG？"},
                {"role": "assistant", "content": "检索增强生成"},
            ],
        },
    )
    assert response.status_code == 200
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 2)
    main._rate_buckets.clear()
    try:
        assert client.get("/api/stats").status_code == 200
        assert client.get("/api/stats").status_code == 200
        limited = client.get("/api/stats")
        assert limited.status_code == 429
        assert "频繁" in limited.json()["detail"]
        # 健康检查不限流
        assert client.get("/api/health").status_code == 200
    finally:
        main._rate_buckets.clear()

def test_retrieve_only(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    client.post(
        "/api/documents",
        json={"name": "note.md", "text": "混合检索 = BM25 + 向量检索，用 RRF 融合排名。"},
    )

    response = client.post(
        "/api/retrieve",
        json={"question": "混合检索怎么融合？", "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hits"], "应该至少命中一个片段"
    assert body["trace"]["candidates"] >= 1
    assert body["latency_ms"] >= 0
    assert body["hits"][0]["doc_name"] == "note.md"


def test_retrieve_empty_collection():
    _ = None
    response = client.post(
        "/api/retrieve",
        json={"question": "随便问问", "collection": "no-such-kb"},
    )
    assert response.status_code == 400
