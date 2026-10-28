"""反馈与指标接口测试：👍/👎 落盘、统计、Prometheus 文本格式。"""

import json

from fastapi.testclient import TestClient

import app.main as main
from app import metrics
from app.store import DocumentStore

client = TestClient(main.app)


def _temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", DocumentStore(tmp_path / "db.json"))
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    metrics.reset()


def test_feedback_is_recorded_and_counted(tmp_path, monkeypatch):
    _temp_store(tmp_path, monkeypatch)
    response = client.post(
        "/api/feedback",
        json={"rating": "up", "question": "什么是 RAG？", "collection": "default"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True

    stats = client.get("/api/feedback/stats").json()
    assert stats["up"] == 1
    assert stats["down"] == 0
    assert stats["total"] == 1
    assert stats["recent"][-1]["question"] == "什么是 RAG？"

    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["rating"] == "up"


def test_feedback_rejects_unknown_rating(tmp_path, monkeypatch):
    _temp_store(tmp_path, monkeypatch)
    response = client.post("/api/feedback", json={"rating": "meh"})
    assert response.status_code == 422


def test_metrics_exposes_prometheus_counters(tmp_path, monkeypatch):
    _temp_store(tmp_path, monkeypatch)

    def fake_answer(question, chunks, top_k=None, history=None):
        return {"answer": "ok", "citations": [], "latency_ms": 12, "tokens": 3, "trace": {}}

    monkeypatch.setattr(main, "rag_answer", fake_answer)
    client.post("/api/ask", json={"question": "你好"})

    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.text
    assert "rag_ask_total 1" in body
    assert "rag_ask_ok 1" in body
    assert "rag_ask_latency_ms_sum 12" in body
    assert "rag_collections" in body
