"""Лучший вариант бонусов (eda/11_meta_boost.md) поверх кандидатов новой гео-схемы hybrid_geo2_{val,bench}.

В score этих кандидатов уже есть гео-бонус (RRF + 0.015 за свой город / 0.02 за крупные города региона),
поэтому сравниваем две базы: "rrf" = 1/(60 + rank) (гео учтено только порядком) и "score" (гео учтено величиной),
и масштаб вектора весов (Вид 0.003, Тип 0.0015, P(mc) 0.007, PRF 0.002) × s. Подбор — на половине A (seed 0),
результат — на половине B. Классификатор: val — train[train_mask], bench — полный train.
Запуск из корня репозитория: python experiments/rerank_meta_geo2.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rerank_meta as rm  # noqa: E402
from rerank_meta_exp import boot, features, recall_series, slice_row  # noqa: E402
from validation import load_validation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, CANDS = ROOT / "data", ROOT / "work" / "cands"
W0 = dict(w_vid=0.003, w_tip=0.0015, w_mc=0.007, w_prf=0.002)


def scaled(s):
    return {k: v * s for k, v in W0.items()}


def main():
    t0 = time.time()
    vq, labels, mask = load_validation()
    items = rm.load_items(DATA / "benchmark_items.parquet")
    tr = pd.read_parquet(DATA / "train.parquet", columns=["search_query", "search_infm_params_text", "item_microcat_id"])
    model = rm.fit_microcat(tr[mask])
    c = pd.read_parquet(CANDS / "hybrid_geo2_val.parquet")
    f = features(c, vq, items, model)
    base = recall_series(c, labels)
    print(f"база geo2: @50 {base.mean():.4f}, @300 {recall_series(c, labels, 300).mean():.4f}")

    qids = np.array(sorted(labels))
    perm = np.random.default_rng(0).permutation(len(qids))
    half_b = set(qids[perm[len(qids) // 2:]])
    ib = base.index.isin(half_b)
    ia = ~ib

    rows, res = [], {}
    for b in ("rrf", "score"):
        for s in (0.5, 1.0, 1.5, 2.0, 3.0):
            r = recall_series(rm.rerank_bonus(f, base=b, **scaled(s)), labels)
            res[(b, s)] = r
            rows.append((b, s, r[ia].mean(), r[ib].mean(), r.mean()))
    grid = pd.DataFrame(rows, columns=["база", "масштаб", "A", "B", "все"])
    print(f"база: A {base[ia].mean():.4f}, B {base[ib].mean():.4f}")
    print(grid.round(4).to_string(index=False))
    bb, bs = grid.loc[grid["A"].idxmax(), ["база", "масштаб"]]
    r = res[(bb, bs)]
    dB, dAll = boot(r[ib], base[ib]), boot(r, base)
    print(f"выбрано по A: base={bb}, масштаб={bs}")
    print(f"B: {base[ib].mean():.4f} -> {r[ib].mean():.4f}, Δ {dB[0]:+.4f} [{dB[1]:+.4f}; {dB[2]:+.4f}]")
    print(f"все: {base.mean():.4f} -> {r.mean():.4f}, Δ {dAll[0]:+.4f} [{dAll[1]:+.4f}; {dAll[2]:+.4f}]")
    reg = vq.set_index("query_id").loc[base.index, "is_region"].to_numpy()
    for name, rr in (("база", base), ("бонус", r)):
        sl = slice_row(rr, vq)
        sl["город"], sl["регион"] = rr[~reg].mean(), rr[reg].mean()
        print(name, "все 3000:", {k: round(v, 4) for k, v in sl.items()})
        print(name, "B:", {k: round(v, 4) for k, v in slice_row(rr[ib], vq).items()})
    new_val = rm.rerank_bonus(f, base=bb, **scaled(bs))
    new_val.to_parquet(CANDS / "hybrid_geo2_meta_val.parquet", index=False)

    del f
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    model_b = rm.fit_microcat(tr)
    cb = pd.read_parquet(CANDS / "hybrid_geo2_bench.parquet")
    nb = rm.rerank_bonus(features(cb, bq, items, model_b), base=bb, **scaled(bs))
    assert nb.groupby("query_id").size().eq(cb.groupby("query_id").size()).all()
    nb.to_parquet(CANDS / "hybrid_geo2_meta_bench.parquet", index=False)
    same = nb[nb["rank"] <= 50].merge(cb[cb["rank"] <= 50], on=["query_id", "item_id"]).shape[0] / (cb["rank"] <= 50).sum()
    print(f"бенчмарк: {nb.query_id.nunique()} запросов, топ-50 совпадает с исходным на {same:.3f}; {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
