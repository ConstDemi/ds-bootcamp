"""Эксперименты с cross-encoder реранкером (eda/12_rerank.md).

  python experiments/rerank_ce_exp.py select            # шаг 1: отбор модели на 800 запросах, топ-100
  python experiments/rerank_ce_exp.py score MODEL K FILE [N]  # досчитать CE-скоры топ-K файла (кеш), N — подвыборка
Скоры кешируются в work/rerank/ce_<модель>_val.parquet, анализ — в rerank_ce_report.py.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
import rerank_ce as R
from validation import load_validation, recall_at_k

CAND = {"geo2": "work/cands/hybrid_geo2_val.parquet", "geo": "work/cands/hybrid_val_geo.parquet"}


def sample_ids(qids, n, seed=0):
    """Подвыборка запросов. n ≤ 800 — вложенные подмножества основной подвыборки 800
    (первые n по query_id), чтобы переиспользовать уже посчитанные скоры."""
    base = set(np.random.default_rng(seed).choice(sorted(qids), 800, replace=False))
    if n >= 800:
        return base if n == 800 else set(np.random.default_rng(seed).choice(sorted(qids), n, replace=False))
    return set(sorted(base)[:n])


def select():
    vq, vl, _ = load_validation()
    ids = sample_ids(vq["query_id"], 800)
    rows = []
    for name, path in CAND.items():
        c = pd.read_parquet(path)
        c = c[c["query_id"].isin(ids)]
        lab = {q: vl[q] for q in ids}
        base = recall_at_k(R.top_lists(c), lab).mean()
        rows.append(dict(cands=name, model="исходный гибрид", r50=base))
        for key in R.MODELS:
            if name != "geo2" and key != "bge-m3":
                continue  # на старых кандидатах — только для сравнения
            torch.cuda.reset_peak_memory_stats()
            cache = R.CACHE / f"ce_{key}_val.parquet"
            t0 = time.time()
            ce = R.ce_scores(c, vq, key, 100, cache=cache, verbose=False)
            dt = time.time() - t0
            for fu in ["ce", "rrf"]:
                r = recall_at_k(R.top_lists(R.fuse(c, ce, 100, fu)), lab).mean()
                rows.append(dict(cands=name, model=key, fusion=fu, r50=r, sec=dt,
                                 vram_gb=torch.cuda.max_memory_allocated() / 2**30))
            print(rows[-2:], flush=True)
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv("work/rerank/select.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


def score(key, k, name, n=None):
    vq, _, _ = load_validation()
    c = pd.read_parquet(CAND[name])
    if n:
        c = c[c["query_id"].isin(sample_ids(vq["query_id"], int(n)))]
    t0 = time.time()
    R.ce_scores(c, vq, key, int(k), cache=R.CACHE / f"ce_{key}_val.parquet")
    print(f"{key} K={k} {name}: {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select()
    else:
        score(*sys.argv[2:])
