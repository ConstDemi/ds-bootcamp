"""Таблицы по экспериментам с географией (work/geo/val_{stem,lemma}.parquet из geo_run.py exp).

Каждое правило считается только на «своих» запросах (региональные правила — на регионах, городские — на городах),
на остальных запросах берётся база C0R0. Сравнение с базой — парный бутстреп по всем 3000 запросам.
Выбор без подгонки: лучшее правило для городов / регионов без 621540 / 621540 выбирается на случайной
половине запросов (seed 0), результат комбинации сообщается на другой половине.

Запуск:  python experiments/geo_report.py [stem|lemma]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geo import RUSSIA  # noqa: E402
from validation import load_validation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "geo"


def boot(d: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load(kind: str):
    vq, vl, _ = load_validation()
    vq["query_id"] = vq["query_id"].astype(str)
    res = pd.read_parquet(OUT / f"val_{kind}.parquet")
    grp = vq.set_index("query_id")
    g = np.where(grp["search_location_id"] == RUSSIA, "russia", np.where(grp["is_region"], "region", "city"))
    grp = pd.Series(g, index=grp.index)
    return vq, res, grp


def full_vector(res, base, grp, rule, col="r50"):
    """Вектор метрики по всем запросам: правило на своих запросах, база на остальных."""
    v = base[col].copy()
    r = res[res["rule"] == rule].set_index("query_id")[col]
    v.loc[r.index] = r
    return v


def table(kind: str):
    vq, res, grp = load(kind)
    base = res[res["rule"] == "C0R0"].set_index("query_id").loc[grp.index]
    rows = []
    for rule in res["rule"].unique():
        v = {c: full_vector(res, base, grp, rule, c) for c in ("r50", "r300")}
        d, lo, hi = boot((v["r50"] - base["r50"]).to_numpy())
        sub = res[res["rule"] == rule].set_index("query_id")
        rows.append({"правило": rule, "все": v["r50"].mean(),
                     "город": v["r50"][grp == "city"].mean(),
                     "регионы без 621540": v["r50"][grp == "region"].mean(),
                     "621540": v["r50"][grp == "russia"].mean(),
                     "@300": v["r300"].mean(), "Δ": d, "ДИ": f"[{lo:+.4f}; {hi:+.4f}]",
                     "в пуле": sub["in_pool"].mean(), "пул, медиана": sub["pool_size"].median()})
    return pd.DataFrame(rows).set_index("правило"), res, base, grp


def choose_half(res, base, grp, seed=0):
    """Выбор правил на половине A, проверка на B (и наоборот)."""
    rng = np.random.default_rng(seed)
    ids = grp.index.to_numpy()
    half_a = set(rng.choice(ids, len(ids) // 2, replace=False))
    in_a = grp.index.isin(half_a)
    out = []
    for name, sel_mask in (("A→B", in_a), ("B→A", ~in_a)):
        chosen = {}
        combo = base["r50"].copy()
        for g in ("city", "region", "russia"):
            rules = res.loc[res["query_id"].isin(grp.index[grp == g]), "rule"].unique()
            m = {}
            for r in rules:
                v = res[res["rule"] == r].set_index("query_id")["r50"]
                idx = grp.index[(grp == g) & sel_mask]
                m[r] = v.reindex(idx).mean()
            best = max(m, key=m.get)
            chosen[g] = best
            v = res[res["rule"] == best].set_index("query_id")["r50"]
            idx = grp.index[grp == g]
            combo.loc[idx] = v.reindex(idx).to_numpy()
        test = ~sel_mask
        d, lo, hi = boot((combo - base["r50"]).to_numpy()[test])
        out.append({"выбор": name, **chosen, "база на тесте": base["r50"][test].mean(),
                    "комбинация на тесте": combo[test].mean(), "Δ": d, "ДИ": f"[{lo:+.4f}; {hi:+.4f}]"})
    return pd.DataFrame(out)


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "stem"
    pd.set_option("display.width", 250)
    tab, res, base, grp = table(kind)
    print(f"[{kind}] запросов: город {int((grp == 'city').sum())}, регионы без 621540 "
          f"{int((grp == 'region').sum())}, 621540 {int((grp == 'russia').sum())}")
    print(tab.round(4).to_string())
    print("\nвыбор на половине, проверка на другой:")
    print(choose_half(res, base, grp).round(4).to_string())
