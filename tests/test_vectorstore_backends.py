"""向量索引后端测试：numpy 精确检索 vs FAISS HNSW 近似检索。

conftest.py 已把 embedding 后端切成哈希向量，所以这里不下载模型、不需要网络。
"""

from __future__ import annotations

import pytest

import app.vectorstore as vectorstore
from app.embeddings import get_embedder
from app.vectorstore import FaissVectorIndex, NumpyVectorIndex, make_index

CHUNKS = [
    {"id": 1, "text": "BM25 是一种词频统计的稀疏检索算法，擅长精确匹配"},
    {"id": 2, "text": "向量检索用 embedding 模型把文本转成向量，能处理同义词"},
    {"id": 3, "text": "红烧肉的做法：五花肉切块，加冰糖生抽慢炖"},
    {"id": 4, "text": "LangGraph 用节点和条件边编排 Agent 流程"},
    {"id": 5, "text": "MySQL 的索引可以加速查询，外键用于保证一致性"},
]


@pytest.fixture(autouse=True)
def clear_index_cache():
    vectorstore.reset_cache()
    yield
    vectorstore.reset_cache()


def test_numpy_search_returns_sorted_hits() -> None:
    index = NumpyVectorIndex(CHUNKS, get_embedder())
    hits = index.search("向量检索 同义词", 2)

    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]
    assert {"id", "text", "score"} <= set(hits[0])


def test_faiss_matches_numpy_top1() -> None:
    pytest.importorskip("faiss", reason="未安装 faiss-cpu，跳过 HNSW 对比")
    embedder = get_embedder()
    numpy_index = NumpyVectorIndex(CHUNKS, embedder)
    faiss_index = FaissVectorIndex(CHUNKS, embedder)

    for question in ["稀疏检索和向量检索区别", "红烧肉要炖多久", "Agent 编排框架", "MySQL 索引"]:
        assert numpy_index.search(question, 1)[0]["id"] == faiss_index.search(question, 1)[0]["id"]


def test_forced_numpy_backend(monkeypatch) -> None:
    monkeypatch.setattr(vectorstore, "VECTOR_BACKEND", "numpy")
    hits = vectorstore.vector_search(CHUNKS, "MySQL 索引", 3)

    assert len(hits) == 3
    assert vectorstore.backend_name() == "numpy"


def test_forced_faiss_backend(monkeypatch) -> None:
    pytest.importorskip("faiss", reason="未安装 faiss-cpu，跳过")
    monkeypatch.setattr(vectorstore, "VECTOR_BACKEND", "faiss")
    hits = vectorstore.vector_search(CHUNKS, "红烧肉", 2)

    assert hits[0]["id"] == 3
    assert vectorstore.backend_name() == "faiss"


def test_empty_chunks_return_no_hits() -> None:
    assert NumpyVectorIndex([], get_embedder()).search("任意问题") == []
    assert FaissVectorIndex([], get_embedder()).search("任意问题") == []


def test_faiss_failure_falls_back_to_numpy(monkeypatch) -> None:
    def boom(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("faiss 不可用")

    monkeypatch.setattr(vectorstore, "FaissVectorIndex", boom)
    index = make_index(CHUNKS, get_embedder(), backend="faiss")

    assert index.backend == "numpy"
    assert index.search("向量检索", 1)[0]["id"] == 2
