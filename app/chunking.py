"""文本切分：先按句子切开，再拼成带重叠的块，保证语义不断。

切分是 RAG 里最容易被忽略、又最影响效果的一步：
- 块太大 → 检索不准、上下文浪费；
- 块太小 → 一句话被切断，模型看不到完整信息；
- 加重叠 → 跨块的句子也能被完整读到。
"""

from __future__ import annotations

import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """把长文本切成若干块。

    参数：
        text: 原始文本
        chunk_size: 每块目标字数（超过就切）
        overlap: 相邻块的重叠字数
    返回：
        非空的块列表；空文本返回 []。
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    overlap = max(0, min(overlap, chunk_size // 2))

    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    pieces: list[str] = []
    buf = ""

    for sent in sentences:
        if len(buf) + len(sent) <= chunk_size or not buf:
            buf += sent
            continue
        pieces.append(buf)
        buf = sent
        # 单个句子就超长：硬切
        step = max(1, chunk_size - overlap)
        while len(buf) > chunk_size:
            pieces.append(buf[:chunk_size])
            buf = buf[step:]

    if buf.strip():
        pieces.append(buf)

    # 相邻块加 overlap 前缀，保证跨块的语义连续
    if overlap and len(pieces) > 1:
        merged = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            merged.append(prev[-overlap:] + cur)
        pieces = merged

    return [p.strip() for p in pieces if p.strip()]
