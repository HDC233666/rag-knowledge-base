from app.chunking import split_text


def test_empty_text_returns_empty_list():
    assert split_text("   \n\n  ") == []


def test_short_text_is_single_chunk():
    chunks = split_text("你好，世界。", chunk_size=600, overlap=80)
    assert len(chunks) == 1
    assert chunks[0] == "你好，世界。"


def test_long_text_is_split_into_multiple_chunks():
    text = "这是第一句话。" * 100
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_overlap_is_applied_between_chunks():
    text = "句子。" * 200
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert chunks[1].startswith(chunks[0][-20:])


def test_chunk_size_zero_raises():
    try:
        split_text("abc", chunk_size=0, overlap=0)
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("chunk_size=0 应该报错")
