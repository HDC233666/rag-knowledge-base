"""多轮会话：SessionStore 单元测试 + 会话接口测试。"""

import time

from fastapi.testclient import TestClient

import app.main as main
from app.session import SessionStore
from app.store import DocumentStore

client = TestClient(main.app)


def _use_temp_store(tmp_path, monkeypatch):
    store = DocumentStore(tmp_path / "db.json")
    monkeypatch.setattr(main, "store", store)
    return store


def _fake_chat(messages, **kwargs):
    # 把历史轮数塞进回答里，方便断言服务端确实带上了上下文
    history = [m for m in messages if m["role"] in {"user", "assistant"}]
    return {"text": f"回答（收到 {len(history)} 条消息）", "latency_ms": 3, "tokens": 20}


def test_store_keeps_only_last_turns():
    store = SessionStore(ttl=60, max_turns=2)
    store.append_turn("u1", "q1", "a1")
    store.append_turn("u1", "q2", "a2")
    store.append_turn("u1", "q3", "a3")
    history = store.history("u1")
    assert len(history) == 4  # 2 轮 × (user + assistant)
    assert history[0]["content"] == "q2"
    assert history[-1]["content"] == "a3"


def test_store_ttl_expires():
    store = SessionStore(ttl=0.05, max_turns=3)
    store.append_turn("u1", "q", "a")
    assert store.history("u1")
    time.sleep(0.12)
    assert store.history("u1") == []
    assert store.get("u1") is None


def test_store_evicts_oldest_session():
    store = SessionStore(ttl=60, max_turns=3, max_sessions=2)
    for name in ("a", "b", "c"):
        store.append_turn(name, "q", "a")
        time.sleep(0.01)
    assert store.stats()["sessions"] == 2
    assert store.get("a") is None  # 最早被挤掉


def test_session_endpoints_roundtrip(tmp_path, monkeypatch):
    _use_temp_store(tmp_path, monkeypatch)
    monkeypatch.setattr("app.rag.llm.chat", _fake_chat)
    client.post("/api/upload", files={"file": ("a.md", "RAG 是检索增强生成。".encode(), "text/markdown")})

    created = client.post("/api/session/new")
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    first = client.post("/api/ask", json={"question": "什么是 RAG？", "session_id": session_id})
    assert first.status_code == 200
    assert first.json()["session_id"] == session_id

    second = client.post("/api/ask", json={"question": "它和微调的区别？", "session_id": session_id})
    assert second.status_code == 200
    # 第二轮应该带上第一轮的问答（服务端记忆，不需要前端传 history）
    assert "收到 3 条消息" in second.json()["answer"]

    got = client.get(f"/api/session/{session_id}")
    assert got.status_code == 200
    assert len(got.json()["turns"]) == 2

    cleared = client.delete(f"/api/session/{session_id}")
    assert cleared.json()["ok"] is True
    assert client.get(f"/api/session/{session_id}").status_code == 404


def test_session_id_validation():
    for bad in ("../etc", "a b"):
        assert client.post("/api/ask", json={"question": "q", "session_id": bad}).status_code == 400
    # 超长 id 在 pydantic 层就被拒（422），规则层还会再兜一次
    assert client.post("/api/ask", json={"question": "q", "session_id": "x" * 65}).status_code == 422
