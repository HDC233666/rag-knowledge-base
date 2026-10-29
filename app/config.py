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

# ===== 检索增强（v2）=====
# 候选池：BM25 和向量各召回多少个，再融合、重排
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "12"))
# RRF 融合常数，论文经验值 60
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
# 查询改写（多查询召回）：关掉可省一次 LLM 调用
ENABLE_REWRITE = os.getenv("RAG_ENABLE_REWRITE", "1").lower() not in {"0", "false", "off", "no"}
# LLM 重排：关掉可省一次 LLM 调用
ENABLE_RERANK = os.getenv("RAG_ENABLE_RERANK", "1").lower() not in {"0", "false", "off", "no"}

# ===== 缓存与重排后端（v3）=====
# 改写 / 重排结果缓存（SQLite + TTL）：重复问题从「秒级」降到「毫秒级」
ENABLE_CACHE = os.getenv("RAG_ENABLE_CACHE", "1").lower() not in {"0", "false", "off", "no"}
CACHE_PATH = os.getenv("RAG_CACHE_PATH", str(DATA_DIR / "cache.db")).strip()
CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL", "3600"))
# 重排后端：llm = LLM 打分（默认）；cross = bge-reranker 交叉编码器（本地 ONNX，便宜、快）
RERANK_BACKEND = os.getenv("RAG_RERANK_BACKEND", "llm").strip().lower()
CROSS_ENCODER_MODEL = os.getenv("RAG_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")

# ===== 防滥用（v3）=====
# 每个 IP 每分钟最多多少个 /api 请求（公网 demo 建议 30~60；0 = 不限制）
RATE_LIMIT_PER_MIN = int(os.getenv("RAG_RATE_LIMIT", "60"))

# ===== 安全与多轮（v6）=====
# 提示注入防护：中和资料里的注入片段 + 输出泄露检查（0 = 关闭）
ENABLE_INJECTION_GUARD = os.getenv("RAG_INJECTION_GUARD", "1").lower() not in {"0", "false", "off", "no"}
# 服务端会话记忆：TTL（秒）与每会话保留的轮数
SESSION_TTL_SECONDS = int(os.getenv("RAG_SESSION_TTL", "1800"))
SESSION_MAX_TURNS = int(os.getenv("RAG_SESSION_MAX_TURNS", "6"))

# ===== 成本估算（v6）=====
# 单价按「元 / 百万 token」，默认取 DeepSeek 公开价：输入（未命中）2.0、输入（命中）0.2、输出 8.0
PRICE_INPUT_PER_M = float(os.getenv("RAG_PRICE_INPUT_PER_M", "2.0"))
PRICE_INPUT_CACHED_PER_M = float(os.getenv("RAG_PRICE_INPUT_CACHED_PER_M", "0.2"))
PRICE_OUTPUT_PER_M = float(os.getenv("RAG_PRICE_OUTPUT_PER_M", "8.0"))

# ===== 向量索引（v4）=====
# auto = 装了 faiss 用 FAISS HNSW，否则 numpy 精确检索；也可强制 numpy / faiss
VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "auto").strip().lower()
# HNSW 每个节点的连边数 M：越大越准、越占内存（一般 16~64）
FAISS_HNSW_M = int(os.getenv("RAG_FAISS_M", "32"))
# 查询时搜索的候选数 efSearch：越大越准、越慢
FAISS_EF_SEARCH = int(os.getenv("RAG_FAISS_EF_SEARCH", "64"))

# ===== 存储后端 =====
# 留空 = 用 JSON 文件（默认，零配置）；填了 = 用 MySQL，连不上会自动降级 JSON
DB_HOST = os.getenv("RAG_DB_HOST", "").strip()
DB_PORT = int(os.getenv("RAG_DB_PORT", "3306"))
DB_USER = os.getenv("RAG_DB_USER", "root").strip()
DB_PASSWORD = os.getenv("RAG_DB_PASSWORD", "")
DB_NAME = os.getenv("RAG_DB_NAME", "rag_kb").strip()
