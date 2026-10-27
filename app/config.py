"""全局配置 —— 全部从环境变量 / .env 读取，改配置不用改代码。"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv 是可选依赖：没装也能跑
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))

# ===== 存储后端 =====
# 留空 = 用 JSON 文件（默认，零配置）；填了 = 用 MySQL，连不上会自动降级 JSON
DB_HOST = os.getenv("RAG_DB_HOST", "").strip()
DB_PORT = int(os.getenv("RAG_DB_PORT", "3306"))
DB_USER = os.getenv("RAG_DB_USER", "root").strip()
DB_PASSWORD = os.getenv("RAG_DB_PASSWORD", "")
DB_NAME = os.getenv("RAG_DB_NAME", "rag_kb").strip()
