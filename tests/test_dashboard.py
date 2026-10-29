"""运行看板：计数、延迟分位数、缓存命中率、成本估算。"""

from fastapi.testclient import TestClient

import app.main as main
from app.store import DocumentStore

client = TestClient(main.app)


def _use_temp_store(tmp_path, monkeypatch):
    store = DocumentStore(tmp_path / "db.json")
    monkeypatch.setattr(main, "store", store)
    return store


def test_latency_stats_percentiles():
    from app import metrics

    metrics.reset()
    for value in range(1, 101):
        metrics.observe_latency("demo", value)
    stats = metrics.latency_stats("demo")
    assert stats["count"] == 100
    assert 50 <= stats["p50"] <= 51
    assert stats["p95"] >= 95
    assert stats["max"] == 100
    assert stats["avg"] == 50.5


def test_cache_hit_rate_and_cost_estimate():
    from app import metrics

    metrics.reset()
    metrics.incr("rag_cache_hit", 3)
    metrics.incr("rag_cache_miss", 1)
    metrics.add_tokens(1_000_000, 0)  # 1M prompt tokens，全按未命中价算
    dashboard = metrics.dashboard()
    assert dashboard["cache"]["hit_rate"] == 0.75
    assert dashboard["cost"]["prompt_tokens"] == 1_000_000
    assert dashboard["cost"]["estimated_cny"] == 2.0


def test_dashboard_endpoint_after_ask(tmp_path, monkeypatch):
    from app import metrics

    metrics.reset()
    _use_temp_store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.rag.llm.chat",
        lambda messages, **kwargs: {
            "text": "RAG 是检索增强生成 [1]",
            "latency_ms": 9,
            "tokens": 40,
            "usage": {"prompt": 1200, "completion": 80, "total": 1280},
        },
    )
    client.post("/api/upload", files={"file": ("a.md", "RAG 是检索增强生成。".encode(), "text/markdown")})
    assert client.post("/api/ask", json={"question": "什么是 RAG？"}).status_code == 200

    data = client.get("/api/dashboard").json()
    assert data["counters"]["rag_ask_ok"] >= 1
    assert data["latency_ms"]["ask"]["count"] >= 1
    assert data["gauges"]["documents"] >= 1
    assert data["cost"]["prompt_tokens"] >= 1200
    assert data["cost"]["estimated_cny"] > 0
    assert "uptime_seconds" in data

    text = client.get("/api/metrics").text
    assert "rag_ask_total" in text
    assert "rag_ask_ms" in text
    assert "rag_estimated_cost_cny" in text
