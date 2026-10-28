"""把 data/corpus/fastapi 下的抓取语料导入知识库（集合 fastapi-docs）。

用法：
    python scripts/ingest_corpus.py            # 导入 data/corpus/fastapi
    python scripts/ingest_corpus.py --dry-run  # 只统计，不写库

幂等：已存在的同名文档会跳过。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import make_store  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus" / "fastapi"
COLLECTION = "fastapi-docs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collection", default=COLLECTION)
    args = parser.parse_args()

    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        print(f"没有找到语料文件: {CORPUS_DIR}")
        return 1

    store = make_store()
    existing = {doc["name"] for doc in store.list_documents(args.collection)}
    print(f"语料文件: {len(files)} 个 | 已入库: {len(existing)} 个 | 存储后端: {type(store).__name__}")

    started = time.perf_counter()
    added = skipped = 0
    for i, path in enumerate(files, 1):
        if path.name in existing:
            skipped += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if args.dry_run:
            added += 1
            continue
        store.add(path.name, text, collection=args.collection)
        added += 1
        if i % 20 == 0:
            print(f"  ... {i}/{len(files)}")

    elapsed = time.perf_counter() - started
    stats = store.stats(args.collection)
    print(
        f"完成：新增 {added}，跳过 {skipped}，耗时 {elapsed:.1f}s\n"
        f"集合「{args.collection}」现在有 {stats['documents']} 个文档 / {stats['chunks']} 个切片"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
