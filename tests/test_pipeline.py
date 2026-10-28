from app import rag
from app import rerank as rerank_mod
from app import rewrite as rewrite_mod

CHUNKS = [
    {"id": "1", "doc_name": "a.md", "index": 0, "text": "RAG 是检索增强生成，先检索资料再让模型生成回答。"},
    {"id": "2", "doc_name": "b.md", "index": 0, "text": "向量检索用 embedding 计算语义相似度。"},
]


def test_rewrite_parses_model_json(monkeypatch):
    monkeypatch.setattr(
        rewrite_mod.llm,
        "chat",
        lambda *args, **kwargs: {"text": '{"queries": ["什么是 RAG", "检索增强生成是什么"]}'},
    )
    queries, used = rewrite_mod.rewrite("什么是 RAG", enabled=True)
    assert used is True
    assert queries[0] == "什么是 RAG"
    assert len(queries) == 2


def test_rewrite_falls_back_when_model_returns_garbage(monkeypatch):
    monkeypatch.setattr(rewrite_mod.llm, "chat", lambda *args, **kwargs: {"text": "抱歉，我不会"})
    queries, used = rewrite_mod.rewrite("什么是 RAG", enabled=True)
    assert used is False
    assert queries == ["什么是 RAG"]


def test_rerank_parses_scores_and_sorts(monkeypatch):
    candidates = [
        {"id": "1", "doc_name": "a.md", "index": 0, "text": "RAG 定义"},
        {"id": "2", "doc_name": "b.md", "index": 0, "text": "向量检索"},
        {"id": "3", "doc_name": "c.md", "index": 0, "text": "红烧肉"},
    ]
    monkeypatch.setattr(
        rerank_mod.llm,
        "chat",
        lambda *args, **kwargs: {
            "text": '{"scores": [{"i": 1, "score": 2}, {"i": 2, "score": 9}, {"i": 3, "score": 1}]}'
        },
    )
    hits, used = rerank_mod.rerank("向量检索是什么", candidates, 2, enabled=True)
    assert used is True
    assert [hit["id"] for hit in hits] == ["2", "1"]
    assert hits[0]["rerank_score"] == 9


def test_rerank_falls_back_on_bad_output(monkeypatch):
    candidates = [
        {"id": "1", "doc_name": "a.md", "index": 0, "text": "RAG 定义"},
        {"id": "2", "doc_name": "b.md", "index": 0, "text": "向量检索"},
    ]
    monkeypatch.setattr(rerank_mod.llm, "chat", lambda *args, **kwargs: {"text": "not json"})
    hits, used = rerank_mod.rerank("问题", candidates, 2, enabled=True)
    assert used is False
    assert [hit["id"] for hit in hits] == ["1", "2"]


def test_answer_returns_citations_and_trace(monkeypatch):
    monkeypatch.setattr(
        rag.llm,
        "chat",
        lambda messages, **kwargs: {
            "text": "RAG 是检索增强生成 [1]",
            "latency_ms": 3,
            "tokens": 20,
        },
    )
    result = rag.answer("什么是 RAG", CHUNKS)
    assert result["citations"]
    assert result["answer"]
    assert "trace" in result
    assert result["trace"]["queries"] == ["什么是 RAG"]


def test_answer_rejects_empty_question():
    try:
        rag.answer("   ", CHUNKS)
    except ValueError as exc:
        assert "不能为空" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("空问题应当抛 ValueError")
