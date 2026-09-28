"""Замеры dense-ретривера на валидации и выгрузка кандидатов для гибрида.

Нужны готовые эмбеддинги корпуса: python src/dense.py encode <модель> --max-seq ... [--tag ...]
Запуск:  python experiments/dense_run.py --model e5-small --tag e5-small [--filters] [--save]

Пишет (с --save) work/cands/dense_{val,bench}_{all,geo}.parquet: query_id, item_id, score, rank (топ-300).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dense import (DATA, ROOT, encode_queries, load_corpus_emb, load_model, search,  # noqa: E402
                   to_frame, to_preds)
from validation import load_validation, recall_at_k, region_location_ids, slices  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="e5-small")
ap.add_argument("--tag", default=None)
ap.add_argument("--max-seq", type=int, default=64)
ap.add_argument("--filters", action="store_true", help="дописывать значения фильтров к запросу")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--save", action="store_true", help="сохранить кандидаты топ-300 в work/cands")
ap.add_argument("--misses", type=int, default=10)
a = ap.parse_args()
torch.set_num_threads(a.threads)
tag = a.tag or a.model
K = 300

emb, ids = load_corpus_emb(tag)
items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_location_id", "item_title_raw"])
assert (items["item_id"].astype(str).to_numpy() == ids).all(), "порядок item_id не совпадает с эмбеддингами"
item_locs = items["item_location_id"].to_numpy()

vq, vl, _ = load_validation()
val_regions = set(vq.loc[vq["is_region"], "search_location_id"])
model = load_model(a.model, a.max_seq)

# --- Валидация ---
t = time.time()
q_emb = encode_queries(model, a.model, vq, with_filters=a.filters)
t_enc = time.time() - t
res = {}
for mode in ("all", "geo"):
    t = time.time()
    res[mode] = search(q_emb, emb, vq, item_locs, val_regions, k=K, mode=mode)
    t_search = time.time() - t
    print(f"{mode}: кодирование запросов {1000 * t_enc / len(vq):.2f} мс/запрос, "
          f"поиск {1000 * t_search / len(vq):.2f} мс/запрос", flush=True)

curve = pd.DataFrame({mode: {k: recall_at_k(to_preds(r, vq, ids, k), vl, k).mean() for k in (50, 100, 200, 300)}
                      for mode, r in res.items()})
print(f"\n[{tag}, filters={a.filters}] Recall@k\n{curve.round(4).to_string()}", flush=True)
for mode, r in res.items():
    print(f"\nСрезы, режим {mode}:\n{slices(recall_at_k(to_preds(r, vq, ids, 50), vl, 50), vq).round(4).to_string()}")

# --- Примеры промахов (режим geo, правильного нет в топ-50) ---
titles = items["item_title_raw"].to_numpy()
pos = {i: n for n, i in enumerate(ids)}
per_q = recall_at_k(to_preds(res["geo"], vq, ids, 50), vl, 50)
miss = per_q[per_q == 0].index.to_series().sample(min(a.misses, int((per_q == 0).sum())), random_state=1)
vqi = vq.set_index("query_id")
print(f"\nПромахи ({(per_q == 0).mean():.1%} запросов без попадания в топ-50, режим geo):")
for qid in miss:
    r = vqi.loc[qid]
    n = vq.index[vq["query_id"] == qid][0]
    gold = [pos[g] for g in vl[qid]]
    rank_all = {i: j for j, (i, _) in enumerate(res["geo"][n], 1)}
    print(f"- «{r.search_query}» [фильтр: {r.search_infm_params_text or '—'}; регион={r.is_region}]\n"
          f"    правильный: {' | '.join(titles[g] for g in gold)} "
          f"(ранг в топ-300: {[rank_all.get(g) for g in gold]}, город совпадает: "
          f"{[bool(item_locs[g] == r.search_location_id) for g in gold]})\n"
          f"    топ-3: {' | '.join(titles[i] for i, _ in res['geo'][n][:3])}")

if a.save:
    out = ROOT / "work" / "cands"
    out.mkdir(parents=True, exist_ok=True)
    for mode, r in res.items():
        to_frame(r, vq, ids).to_parquet(out / f"dense_val_{mode}.parquet", index=False)
    # --- Бенчмарк: тот же код, регионы — локации без объявлений ---
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    b_emb = encode_queries(model, a.model, bq, with_filters=a.filters)
    regions = region_location_ids()
    for mode in ("all", "geo"):
        r = search(b_emb, emb, bq, item_locs, regions, k=K, mode=mode)
        df = to_frame(r, bq, ids)
        assert df.groupby("query_id").size().eq(K).all() and df["query_id"].nunique() == len(bq)
        df.to_parquet(out / f"dense_bench_{mode}.parquet", index=False)
    print("кандидаты сохранены в", out)
