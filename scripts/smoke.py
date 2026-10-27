"""冒烟测试：用真实 Key 跑通「导入示例文档 → 提问 → 得到带引用的答案」。

用法（项目根目录、已激活 venv、已配置 DEEPSEEK_API_KEY）：
    python scripts/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台默认 GBK，统一成 UTF-8，避免中文/符号报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def main() -> None:
    with TestClient(app) as client:
        stats = client.get("/api/stats").json()
        print("知识库统计:", stats)
        if not stats["documents"]:
            raise SystemExit("没有文档：data/sample 下的示例文档没被导入")
        response = client.post("/api/ask", json={"question": "什么是 RAG？"})
        if response.status_code != 200:
            raise SystemExit(f"提问失败：{response.status_code} {response.text}")
        data = response.json()
        print("回答:", data["answer"][:200].replace("\n", " "))
        print("引用数:", len(data["citations"]), "| 耗时:", data["latency_ms"], "ms")
        if not data["citations"]:
            raise SystemExit("回答没有引用，检索链路可能有问题")
        print("OK")


if __name__ == "__main__":
    main()
