"""访问保护：口令与每日预算（DEMO_PASSWORD / DEMO_DAILY_BUDGET）。"""

from pathlib import Path

from fastapi.testclient import TestClient
from starlette.requests import Request

from app import guard
from app.main import app


def _request(query: str = "", cookie: str | None = None) -> Request:
    headers = []
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/documents",
            "headers": headers,
            "query_string": query.lstrip("?").encode(),
        }
    )


def test_disabled_means_unlocked(monkeypatch):
    monkeypatch.setattr(guard, "PASSWORD", "")
    assert guard.auth_enabled() is False
    assert guard.is_unlocked(_request()) is True
    assert guard.budget_left() == -1


def test_password_cookie_and_query(monkeypatch):
    monkeypatch.setattr(guard, "PASSWORD", "demo-pw")
    digest = guard._digest("demo-pw")
    assert guard.auth_enabled() is True
    assert guard.is_unlocked(_request()) is False
    assert guard.is_unlocked(_request("?k=demo-pw")) is True
    assert guard.is_unlocked(_request("?k=wrong")) is False
    assert guard.is_unlocked(_request(cookie=f"demo_pass={digest}")) is True
    assert guard.is_unlocked(_request(cookie="demo_pass=bad")) is False


def test_budget_counter(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(guard, "_STATE", tmp_path / "guard.json")
    monkeypatch.setattr(guard, "DAILY_BUDGET", 2)
    assert guard.budget_left() == 2
    assert guard.check_budget() is True
    assert guard.check_budget() is True
    assert guard.check_budget() is False
    assert guard.budget_left() == 0


def test_unlock_endpoint_sets_cookie(monkeypatch):
    monkeypatch.setattr(guard, "PASSWORD", "demo-pw")
    client = TestClient(app)
    assert client.get("/api/documents").status_code == 401
    assert client.post("/api/unlock", json={"password": "bad"}).status_code == 403
    ok = client.post("/api/unlock", json={"password": "demo-pw"})
    assert ok.status_code == 200
    assert ok.json()["enabled"] is True
    assert client.get("/api/documents").status_code == 200
    # 页面本身不用口令
    assert client.get("/").status_code == 200


def test_health_is_public(monkeypatch):
    monkeypatch.setattr(guard, "PASSWORD", "demo-pw")
    client = TestClient(app)
    body = client.get("/api/health").json()
    assert body["auth"] is True
    assert client.get("/api/stats").status_code == 401

