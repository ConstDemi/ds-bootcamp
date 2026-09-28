"""Отчёт по итоговой гео-конфигурации (после `geo_run.py final`): срезы, кривая, потери пула, промахи регионов.

Запуск:  python experiments/geo_final_report.py lemma
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geo import RUSSIA, centroids, load_geo_data  # noqa: E402
from geo_report import boot  # noqa: E402
from validation import load_validation, slices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "geo"
CANDS = ROOT / "work" / "cands"

kind = sys.argv[1] if len(sys.argv) > 1 else "lemma"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
vq, vl, mask = load_validation()
vq["query_id"] = vq["query_id"].astype(str)
fin = pd.read_parquet(OUT / f"final_val_{kind}.parquet").set_index("query_id").loc[vq["query_id"]]
old = pd.read_parquet(OUT / "val_stem.parquet")
old = old[old["rule"] == "C0R0"].set_index("query_id").loc[vq["query_id"]]

print("правила по типам запросов:", fin.groupby(vq.set_index("query_id")["is_region"].to_numpy())["rule"]
      .agg(lambda s: s.value_counts().to_dict()).to_dict())
print("\nкривая Recall@k:")
print(pd.DataFrame({"было (C0R0, stem)": old[["r50", "r100", "r200", "r300"]].mean(),
                    "стало": fin[["r50", "r100", "r200", "r300"]].mean()}).T.round(4).to_string())
d, lo, hi = boot((fin["r50"] - old["r50"]).to_numpy())
print(f"Δ@50 = {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")
sl = pd.concat({"было": slices(old["r50"], vq)["recall"], "стало": slices(fin["r50"], vq)["recall"]}, axis=1)
sl.insert(0, "запросов", slices(fin["r50"], vq)["запросов"].astype(int))
sl["Δ"] = sl["стало"] - sl["было"]
print("\nсрезы Recall@50:\n", sl.round(4).to_string(), sep="")
g = vq.set_index("query_id")
typ = np.where(g["search_location_id"] == RUSSIA, "621540", np.where(g["is_region"], "регион", "город"))
print("\nпо типу:", pd.DataFrame({"было": old["r50"].groupby(typ).mean(), "стало": fin["r50"].groupby(typ).mean(),
                                   "в пуле ярусов": fin["in_pool"].groupby(typ).mean(),
                                   "пул, медиана": fin["pool_size"].groupby(typ).median()}).round(4).to_string())
print("потери пула (доля правильных вне ярусов до добора), все:", round(1 - fin["in_pool"].fillna(1).mean(), 4))

# ---- промахи по регионам
corpus, train = load_geo_data()
tr_fit = train[mask]
items = pd.read_parquet(ROOT / "data" / "benchmark_items.parquet", columns=["item_id", "item_title_raw"])
title = dict(zip(items["item_id"].astype(str), items["item_title_raw"].str.slice(0, 55)))
iloc = dict(zip(corpus["item_id"], corpus["item_location_id"]))
cent = centroids(corpus, train)
reg = tr_fit[tr_fit["search_location_id"].isin(set(g.loc[g["is_region"], "search_location_id"]))]
share = reg.groupby(["search_location_id", "item_location_id"]).size()
share = share / share.groupby(level=0).transform("sum")
cand = pd.read_parquet(CANDS / "hybrid_geo2_val.parquet")
rank = cand.set_index(["query_id", "item_id"])["rank"]
top = cand[cand["rank"] <= 3].groupby("query_id")["item_id"].agg(list)
miss = fin[(fin["r50"] == 0) & g["is_region"].to_numpy()].index
print(f"\nпромахов среди регионов: {len(miss)} из {int(g['is_region'].sum())}")
for q in pd.Series(miss).sample(min(10, len(miss)), random_state=3):
    rel = sorted(vl[q])[0]
    sl_ = g.at[q, "search_location_id"]
    il = iloc[rel]
    sh = share.get((sl_, il), 0.0)
    main = share.loc[sl_].idxmax() if sl_ in share.index.get_level_values(0) else None
    dist = np.nan
    if main is not None and il in cent.index and main in cent.index:
        from geo import hav
        dist = hav(cent.at[il, "clat"], cent.at[il, "clon"], cent.at[main, "clat"], cent.at[main, "clon"])
    print(f"- «{g.at[q, 'search_query']}» [{g.at[q, 'search_infm_params_text'] or ''}] loc={sl_} | "
          f"ответ: {title[rel]} (loc {il}, доля в регионе {sh:.3%}, до главного города {dist:.0f} км) | "
          f"ранг: {rank.get((q, rel), '—')} | топ-3: {'; '.join(title[i] for i in top[q])}")
