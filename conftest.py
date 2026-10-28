"""让 pytest 在项目根目录也能 import app 包（Windows / macOS 通用）。

同时把测试环境的开关关掉：
- 向量检索用哈希兜底实现（不下载模型、CI 也能跑）；
- 查询改写 / LLM 重排默认关闭（测试不花钱、不依赖网络）。
想测真实模型时，在命令行前面加 RAG_EMBED_BACKEND=auto RAG_ENABLE_REWRITE=1 覆盖。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_EMBED_BACKEND", "hash")
os.environ.setdefault("RAG_ENABLE_REWRITE", "0")
os.environ.setdefault("RAG_ENABLE_RERANK", "0")
os.environ.setdefault("RAG_ENABLE_CACHE", "0")
os.environ.setdefault("RAG_RATE_LIMIT", "0")
