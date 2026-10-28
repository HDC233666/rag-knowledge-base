"""LLM 调用结果缓存（SQLite + TTL）。

改写 / 重排这类调用有个特点：同样的输入，输出基本不变，却每次都要花一次 LLM 调用。
把它们的结果缓存起来，重复问题就能从「秒级」降到「毫秒级」——面试里讲的
「缓存把 P95 延迟从 1.4s 降到 40ms」就是这么来的。

实现刻意做得很小：
- 一张表：key / value(JSON) / expires_at；
- 读写失败全部静默（缓存只是优化，绝不能影响主流程）；
- 过期的行不主动清理，命中时判断过期即可（规模小，无所谓）。

配置：RAG_ENABLE_CACHE=0 关闭；RAG_CACHE_TTL 控制秒数；RAG_CACHE_PATH 换文件位置。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from . import config


def _connect() -> sqlite3.Connection:
    path = Path(config.CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=2)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at REAL NOT NULL)"
    )
    return conn


def _key(namespace: str, text: str) -> str:
    return hashlib.sha256(f"{namespace}\x00{text}".encode("utf-8")).hexdigest()


def get(namespace: str, text: str) -> dict | None:
    """命中且未过期则返回缓存值（dict），否则 None。"""
    if not config.ENABLE_CACHE:
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
                (_key(namespace, text), time.time()),
            ).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:  # noqa: BLE001
        return None


def put(namespace: str, text: str, value: dict, ttl: int | None = None) -> None:
    """写入缓存（覆盖同 key）；任何异常都吞掉。"""
    if not config.ENABLE_CACHE:
        return
    ttl = config.CACHE_TTL_SECONDS if ttl is None else ttl
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO cache(key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at",
                (_key(namespace, text), json.dumps(value, ensure_ascii=False), time.time() + ttl),
            )
    except Exception:  # noqa: BLE001
        pass


def stats() -> dict:
    """给接口/文档用：缓存条数。"""
    try:
        with _connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        return {"entries": int(total), "path": config.CACHE_PATH, "ttl_seconds": config.CACHE_TTL_SECONDS}
    except Exception:  # noqa: BLE001
        return {"entries": 0, "path": config.CACHE_PATH, "ttl_seconds": config.CACHE_TTL_SECONDS}
