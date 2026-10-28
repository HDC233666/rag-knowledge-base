from app.retrieval import hybrid_search, rrf_fuse

CHUNKS = [
    {"id": "1", "doc_name": "a.md", "index": 0, "text": "RAG 是检索增强生成，先检索资料再让模型生成回答。"},
    {"id": "2", "doc_name": "b.md", "index": 0, "text": "向量检索用 embedding 计算语义相似度。"},
    {"id": "3", "doc_name": "c.md", "index": 0, "text": "红烧肉的做法：五花肉加冰糖，小火慢炖两小时。"},
]


def test_rrf_fuses_by_rank_not_by_raw_score():
    bm25_like = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    vector_like = [{"id": "2"}, {"id": "3"}, {"id": "1"}]
    fused = rrf_fuse([bm25_like, vector_like])
    # 2 在两路里分别排第 2、第 1，融合分最高
    assert fused[0]["id"] == "2"
    # 融合分 = 1/(60+2) + 1/(60+1)（rrf 字段保留 6 位小数）
    assert abs(fused[0]["rrf"] - (1 / 62 + 1 / 61)) < 1e-6


def test_rrf_keeps_single_list_order():
    fused = rrf_fuse([[{"id": "a"}, {"id": "b"}]])
    assert [item["id"] for item in fused] == ["a", "b"]


def test_hybrid_search_returns_fused_hits_with_sources():
    hits = hybrid_search(CHUNKS, "RAG 是什么", top_k=2)
    assert hits
    assert "rrf" in hits[0]
    assert "sources" in hits[0]
    assert hits[0]["doc_name"] in {"a.md", "b.md", "c.md"}


def test_hybrid_search_empty_corpus():
    assert hybrid_search([], "RAG") == []
