"""Справляется ли готовый эмбеддер без дообучения: кривая Recall@K до K=1000 и точное место
правильного объявления среди всего корпуса (и для сравнения то же для BM25).

Запуск: python experiments/rank_curve.py   (нужны work/emb/bge-m3_s256.* и GPU; ~3 мин)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dense import DATA, encode_queries, load_corpus_emb, load_model, search, to_preds  # noqa: E402
from sparse import SparseIndex, load_items  # noqa: E402
from sparse import to_preds as sparse_preds  # noqa: E402
from validation import load_validation, recall_at_k, slices  # noqa: E402

KS = (50, 100, 200, 300, 500, 1000)

vq, vl, _ = load_validation()
regions = set(vq.loc[vq["is_region"], "search_location_id"])

# --- Dense: bge-m3, запрос + значения фильтров (как в лучшем варианте) ---
emb, ids = load_corpus_emb("bge-m3_s256")
items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_location_id"])
assert (items["item_id"].astype(str).to_numpy() == ids).all()
item_locs = items["item_location_id"].to_numpy()
model = load_model("bge-m3", 256)
q_emb = encode_queries(model, "bge-m3", vq, with_filters=True)

dense = {m: search(q_emb, emb, vq, item_locs, regions, k=max(KS), mode=m) for m in ("all", "geo")}
curve = {f"dense {m}": {k: recall_at_k(to_preds(r, vq, ids, k), vl, k).mean() for k in KS}
         for m, r in dense.items()}

# Точное место правильного объявления среди всех 189 тыс. (1 = первое). Если правильных несколько — лучшее.
pos = {i: n for n, i in enumerate(ids)}
D = torch.from_numpy(emb).cuda().half()
Q = torch.from_numpy(q_emb).cuda().half()
dense_rank = {}
for n, qid in enumerate(vq["query_id"]):
    s = D @ Q[n]
    gold = torch.tensor([pos[g] for g in vl[qid]], device="cuda")
    dense_rank[qid] = int((s[None, :] > s[gold][:, None]).sum(1).min().item()) + 1

# --- Sparse: BM25 со стеммингом, заголовок ×2 (вариант b) ---
sp_items = load_items()
index = SparseIndex.build(sp_items, variant="b")
sparse = {m: index.search(vq, mode=m, k=max(KS)) for m in ("all", "geo")}
for m, c in sparse.items():
    curve[f"bm25 {m}"] = {k: recall_at_k(sparse_preds(c, k), vl, k).mean() for k in KS}
sp_pos = {i: n for n, i in enumerate(index.item_ids)}
sparse_rank = {}
for qid, text in zip(vq["query_id"], vq["search_query"]):
    sc = index.scores(text)
    g = np.array([sp_pos[x] for x in vl[qid]])
    sparse_rank[qid] = int((sc[None, :] > sc[g][:, None]).sum(1).min()) + 1

print("Recall@K на валидации (3000 запросов)")
print(pd.DataFrame(curve).round(4).to_string())

ranks = pd.DataFrame({"dense": dense_rank, "bm25": sparse_rank})
q = [0.25, 0.5, 0.75, 0.9]
print("\nМесто правильного объявления во всём корпусе (189 212), квантили по запросам")
print(ranks.quantile(q).astype(int).to_string())
bins = [0, 50, 300, 1000, 10000, 10**9]
labels = ["1–50", "51–300", "301–1000", "1001–10000", ">10000"]
print("\nДоля запросов по месту правильного объявления во всём корпусе")
print(ranks.apply(lambda s: pd.cut(s, bins, labels=labels).value_counts(normalize=True).sort_index())
      .mul(100).round(1).to_string())

print("\nСрезы dense, Recall@500 (all / geo)")
print(pd.concat({m: slices(recall_at_k(to_preds(r, vq, ids, 500), vl, 500), vq)["recall"]
                 for m, r in dense.items()}, axis=1).round(4).to_string())
