"""极简指标收集：进程内计数器，供 /api/metrics 以 Prometheus 文本格式暴露。

不引第三方库：多线程安全靠一把锁，够小项目用。
生产环境可换成 prometheus_client + 独立进程，但这里保留「零依赖」的取舍。
"""

from __future__ import annotations

import threading
from collections import Counter

_lock = threading.Lock()
_counters: Counter[str] = Counter()
_gauges: dict[str, float] = {}


def incr(name: str, value: int = 1) -> None:
    with _lock:
        _counters[name] += value


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _gauges[name] = float(value)


def snapshot() -> dict:
    with _lock:
        return {"counters": dict(_counters), "gauges": dict(_gauges)}


def reset() -> None:
    """测试用：清空所有指标。"""
    with _lock:
        _counters.clear()
        _gauges.clear()


def render_prometheus() -> str:
    """输出 Prometheus 文本格式：每个指标一行 # TYPE 可选。"""
    data = snapshot()
    lines: list[str] = []
    for name, value in sorted(data["counters"].items()):
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")
    for name, value in sorted(data["gauges"].items()):
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value:g}")
    if not lines:
        lines.append("# no metrics yet")
    return "\n".join(lines) + "\n"
