"""临时诊断：看 hybrid 在哪些题上把正确文档挤掉了，并比较不同融合参数。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.chunking import split_text
from app.config import DATA_DIR
from app.embeddings import get_embedder
from app.retrieval import search as bm25_search
from app.vectorstore import vector_search

payload = json.loads((DATA_DIR / "eval" / "questions.json").read_text(encoding="utf-8"))
questions = payload["questions"] if isinstance(payload, dict) else payload
docs = []
for path in sorted((DATA_DIR / "sample").glob("*.md")):
    docs.append((path.name, path.read_text(encoding="utf-8")))
chunks = []
for name, text in docs:
    for i, piece in enumerate(split_text(text)):
        chunks.append({"id": f"{name}#{i}", "doc_name": name, "index": i, "text": piece})
print(f"chunks={len(chunks)} questions={len(questions)}")
embedder = get_embedder()
print("embedder:", embedder.name)


def hit_rank(list_, target):
    for rank, item in enumerate(list_, start=1):
        if item["doc_name"] == target:
            return rank
    return None


K = 60
for ck in (6, 8, 10, 12, 14):
    for w in (0.6, 0.8, 1.0):
        hits1 = hits3 = 0
        mrr = 0.0
        for q in questions:
            bm = bm25_search(chunks, q["question"], top_k=ck)
            vec = vector_search(chunks, q["question"], top_k=ck, embedder=embedder)
            fused = {}
            for items, weight in ((bm, 1.0), (vec, w)):
                for rank, item in enumerate(items, start=1):
                    fused[item["id"]] = fused.get(item["id"], 0.0) + weight / (K + rank)
            order = [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
            by_id = {c["id"]: c for c in chunks}
            ranked = [by_id[cid] for cid in order]
            r = hit_rank(ranked, q["doc"])
            if r == 1:
                hits1 += 1
            if r and r <= 3:
                hits3 += 1
            if r:
                mrr += 1 / r
        print(f"candidate_k={ck} w_vec={w}: hit@1={hits1/len(questions):.2f} hit@3={hits3/len(questions):.2f} MRR={mrr/len(questions):.3f}")

