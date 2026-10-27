"""把上传的文件 / 本地路径变成纯文本。支持 txt、md、pdf、csv、log。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".csv", ".log"}


def _pdf_to_text(stream) -> str:
    reader = PdfReader(stream)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n".join(page for page in pages if page)


def load_bytes(data: bytes, filename: str) -> str:
    """从上传的字节流提取文本。"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _pdf_to_text(BytesIO(data))
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"不支持的文件类型 {suffix or '(无后缀)'}，"
            f"支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    return data.decode("utf-8", errors="replace")


def load_text(path: Path) -> str:
    """从本地路径读取文本（PDF 自动抽取）。"""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _pdf_to_text(str(path))
    return path.read_text(encoding="utf-8", errors="replace")
