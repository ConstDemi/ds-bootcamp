"""Быстрое сравнение моделей эмбеддингов на подкорпусе (без GPU полный корпус для каждой модели кодировать долго).

Подкорпус = все правильные объявления валидации + N случайных объявлений-«отвлекающих».
Метрика — Recall@{1,10,50} по валидации в режиме all. Абсолютные числа выше, чем на полном корпусе,
но порядок моделей/настроек сохраняется: все варианты ищут в одном и том же подкорпусе.

Запуск: python experiments/dense_compare.py --threads 3 --distractors 5000
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
from dense import DATA, ITEM_COLS, MODELS, encode_queries, load_model, search, to_preds  # noqa: E402
from text import item_text  # noqa: E402
from validation import load_validation, recall_at_k  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--distractors", type=int, default=5000)
ap.add_argument("--configs", default="e5-small:64:q,e5-small:64:fp32,e5-small:128:q,user-base:64:q")
a = ap.parse_args()
torch.set_num_threads(a.threads)

vq, vl, _ = load_validation()
items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS)
gold = set().union(*vl.values())
rng = np.random.default_rng(0)
other = np.flatnonzero(~items["item_id"].isin(gold).to_numpy())
sub = pd.concat([items[items["item_id"].isin(gold)],
                 items.iloc[rng.choice(other, a.distractors, replace=False)]]).reset_index(drop=True)
ids = sub["item_id"].astype(str).to_numpy()
print(f"подкорпус: {len(sub)} объявлений ({len(gold)} правильных)", flush=True)

rows = []
for cfg in a.configs.split(","):
    key, L, qmode = cfg.split(":")
    m = load_model(key, int(L), quantize=(qmode == "q"))
    t = time.time()
    d = m.encode((MODELS[key]["d"] + item_text(sub)).tolist(), batch_size=64, prompt="",
                 normalize_embeddings=True, convert_to_numpy=True)
    t_doc = (time.time() - t) / len(sub) * 1000
    for wf in (False, True):
        q = encode_queries(m, key, vq, with_filters=wf)
        res = search(q, d, vq, sub["item_location_id"].to_numpy(), set(), k=50, mode="all")
        r = {f"R@{k}": recall_at_k(to_preds(res, vq, ids, k), vl, k).mean() for k in (1, 10, 50)}
        rows.append(dict(config=cfg, filters=wf, ms_per_doc=round(t_doc, 2), **r))
        print(rows[-1], flush=True)
print(pd.DataFrame(rows).round(4).to_string(index=False))
