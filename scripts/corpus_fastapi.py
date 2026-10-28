"""抓取 FastAPI 中文文档，落成本地 txt 语料，供 RAG 大规模检索评估使用。

用法：
    python scripts/corpus_fastapi.py --limit 120

产物：
    data/corpus/fastapi/<slug>.txt

只依赖 httpx（项目已有）。脚本可重复运行，已存在的文件会跳过。
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path

import httpx

BASE = "https://fastapi.tiangolo.com"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus" / "fastapi"

TAG_RE = re.compile(r"<[^>]+>")
ARTICLE_RE = re.compile(r'<article[^>]*>(.*?)</article>', re.S)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)


def sitemap_urls(client: httpx.Client) -> list[str]:
    resp = client.get(f"{BASE}/zh/sitemap.xml")
    resp.raise_for_status()
    urls = re.findall(r"<loc>(.*?)</loc>", resp.text)
    return [u for u in urls if "/zh/" in u]


def html_to_text(page: str) -> str:
    match = ARTICLE_RE.search(page)
    body = match.group(1) if match else page
    body = re.sub(r"<(script|style|nav|footer)\b.*?</\1>", " ", body, flags=re.S)
    body = re.sub(r"<(/?)(h[1-6]|p|li|pre|tr|div|section|br)[^>]*>", "\n", body, flags=re.I)
    body = TAG_RE.sub(" ", body)
    text = html.unescape(body)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def slug_of(url: str) -> str:
    path = url.replace(BASE, "").strip("/")
    if not path:
        return "index"
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", path).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120, help="最多抓多少页（默认 120）")
    parser.add_argument("--min-chars", type=int, default=300, help="正文少于此值跳过")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"user-agent": "Mozilla/5.0 (corpus-builder; RAG demo)"}
    with httpx.Client(timeout=25, follow_redirects=True, headers=headers) as client:
        urls = sitemap_urls(client)
        print(f"发现 {len(urls)} 个中文页面")
        ok = skip = fail = 0
        for i, url in enumerate(urls[: args.limit], 1):
            slug = slug_of(url)
            target = OUT_DIR / f"{slug}.txt"
            if target.exists() and target.stat().st_size > 200:
                skip += 1
                continue
            try:
                resp = client.get(url)
                resp.raise_for_status()
                title = TITLE_RE.search(resp.text)
                title_text = html.unescape(title.group(1)).strip() if title else slug
                text = html_to_text(resp.text)
                if len(text) < args.min_chars:
                    skip += 1
                    continue
                target.write_text(f"{title_text}\n\n{text}\n", encoding="utf-8")
                ok += 1
            except Exception as exc:  # noqa: BLE001 - 单页失败不影响整体
                fail += 1
                print(f"  [warn] {url} -> {type(exc).__name__}: {exc}")
            if i % 20 == 0:
                print(f"  ... {i}/{min(len(urls), args.limit)}（ok={ok} skip={skip} fail={fail}）")
                time.sleep(0.2)

        print(f"完成：新增 {ok}，跳过 {skip}，失败 {fail}；目录 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
