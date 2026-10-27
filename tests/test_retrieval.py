from app.retrieval import search, tokenize

CHUNKS = [
    {"id": "1", "doc_name": "a.md", "index": 0, "text": "RAG 是检索增强生成，先检索资料再让模型生成回答。"},
    {"id": "2", "doc_name": "b.md", "index": 0, "text": "BM25 是一种传统的文本检索算法，按关键词打分。"},
    {"id": "3", "doc_name": "c.md", "index": 0, "text": "红烧肉的做法：五花肉加冰糖，小火慢炖两小时。"},
]


def test_search_returns_most_relevant_chunk_first():
    hits = search(CHUNKS, "RAG 是什么", top_k=2)
    assert hits
    assert hits[0]["doc_name"] == "a.md"
    assert "score" in hits[0]


def test_search_empty_corpus_returns_empty():
    assert search([], "RAG") == []


def test_search_empty_query_returns_empty():
    assert search(CHUNKS, "   ") == []


def test_tokenize_handles_chinese_and_english():
    tokens = tokenize("RAG 检索")
    assert "rag" in tokens
    assert "检索" in tokens
