"""服务端多轮会话记忆（进程内，带 TTL）。

为什么要有
    前端自己传 history 也能多轮，但那样有两个问题：
    1. 客户端可以随便伪造历史（把别人的对话塞进来）；
    2. 每次请求都要把整段历史带上，长对话越来越慢、越来越贵。
    改成服务端会话：前端只发 session_id，服务端存最近 N 轮，超时自动清。
    面试里可以讲成「会话隔离 + 上下文裁剪 + 内存上限」三件事。

实现取舍
    - 进程内 OrderedDict：单机 demo 够用，重启即失效；要持久化就换 Redis/MySQL；
    - 每个会话只保留最近 max_turns 轮（默认 6），注入 prompt 时再截到 3 轮；
    - TTL 默认 30 分钟，访问时懒清理 + 超过 max_sessions 时淘汰最久未用的。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

from . import config


class SessionStore:
    """线程安全的小型会话表。"""

    def __init__(self, ttl: int | None = None, max_turns: int | None = None, max_sessions: int = 2000):
        self.ttl = config.SESSION_TTL_SECONDS if ttl is None else ttl
        self.max_turns = config.SESSION_MAX_TURNS if max_turns is None else max_turns
        self.max_sessions = max_sessions
        self._lock = threading.RLock()
        self._data: OrderedDict[str, dict] = OrderedDict()

    # ---------- 内部 ----------
    def _purge(self, now: float) -> None:
        expired = [sid for sid, item in self._data.items() if now - item["updated_at"] > self.ttl]
        for sid in expired:
            self._data.pop(sid, None)
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)

    def _bucket(self, session_id: str, now: float) -> dict:
        item = self._data.get(session_id)
        if item is None:
            item = {"turns": [], "created_at": now, "updated_at": now}
            self._data[session_id] = item
        item["updated_at"] = now
        self._data.move_to_end(session_id)
        return item

    # ---------- 对外 ----------
    def history(self, session_id: str) -> list[dict]:
        """最近 N 轮的扁平消息列表 [{role, content}]，按时间正序。"""
        if not session_id:
            return []
        now = time.time()
        with self._lock:
            self._purge(now)
            item = self._data.get(session_id)
            if item is None or now - item["updated_at"] > self.ttl:
                return []
            messages: list[dict] = []
            for turn in item["turns"][-self.max_turns :]:
                messages.append({"role": "user", "content": turn["question"]})
                if turn.get("answer"):
                    messages.append({"role": "assistant", "content": turn["answer"]})
            return messages

    def append_turn(self, session_id: str, question: str, answer: str = "") -> None:
        if not session_id or not question:
            return
        now = time.time()
        with self._lock:
            self._purge(now)
            item = self._bucket(session_id, now)
            item["turns"].append(
                {
                    "question": question[:2000],
                    "answer": (answer or "")[:4000],
                    "at": now,
                }
            )
            item["turns"] = item["turns"][-self.max_turns :]

    def get(self, session_id: str) -> dict | None:
        """调试用：返回该会话的原始记录（含时间戳）。"""
        with self._lock:
            item = self._data.get(session_id)
            if item is None:
                return None
            return {
                "session_id": session_id,
                "turns": list(item["turns"]),
                "created_at": item["created_at"],
                "updated_at": item["updated_at"],
            }

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._data.pop(session_id, None) is not None

    def stats(self) -> dict:
        now = time.time()
        with self._lock:
            self._purge(now)
            turns = sum(len(item["turns"]) for item in self._data.values())
            return {
                "sessions": len(self._data),
                "turns": turns,
                "ttl_seconds": self.ttl,
                "max_turns": self.max_turns,
            }

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]


sessions = SessionStore()
