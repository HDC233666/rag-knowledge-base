"""缓存测试：读写、过期、关闭开关。全部用临时 SQLite 文件，不碰真实缓存。"""

from __future__ import annotations

from app import cache, config


def test_cache_roundtrip(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "ENABLE_CACHE", True)
    monkeypatch.setattr(config, "CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setattr(config, "CACHE_TTL_SECONDS", 60)

    assert cache.get("ns", "abc") is None
    cache.put("ns", "abc", {"queries": ["a", "b"]})
    assert cache.get("ns", "abc") == {"queries": ["a", "b"]}


def test_cache_expired(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "ENABLE_CACHE", True)
    monkeypatch.setattr(config, "CACHE_PATH", str(tmp_path / "cache.db"))

    cache.put("ns", "abc", {"v": 1}, ttl=0)
    assert cache.get("ns", "abc") is None


def test_cache_disabled(monkeypatch) -> None:
    monkeypatch.setattr(config, "ENABLE_CACHE", False)
    cache.put("ns", "abc", {"v": 1})
    assert cache.get("ns", "abc") is None
