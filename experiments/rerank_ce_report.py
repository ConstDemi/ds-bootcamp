"""Анализ CE-реранка по закешированным скорам (eda/12_rerank.md).

  python experiments/rerank_ce_report.py MODEL CANDS_NAME K1,K2,... [N_SAMPLE]
CANDS_NAME: geo2 | geo (см. rerank_ce_exp.CAND). N_SAMPLE — оценка на подвыборке (seed 0), если
для всей валидации нет скоров больших K.

Сетка слияний: ce; rrf (rrf_k=60, w_ce ∈ {1, 2}); keep (m ∈ {20, 30}).
Параметры (K и слияние) выбираются на половине A валидации (случайная, seed 0), результат — на половине B.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import rerank_ce as R
from rerank_ce_exp import CAND, sample_ids
from validation import load_validation, recall_at_k, slices


def bootstrap_diff(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0):
    """Парный бутстреп по запросам: среднее (a − b) и 95% интервал (как в hybrid.py)."""
    d = (a - b.loc[a.index]).to_numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def grid(ks):
    for k in ks:
        yield dict(k=k, fusion="ce")
        for w in (1.0, 2.0):
            yield dict(k=k, fusion="rrf", w_ce=w)
        for m in (20, 30):
            yield dict(k=k, fusion="keep", m=m)


def name(p):
    s = f"K={p['k']} {p['fusion']}"
    if p["fusion"] == "rrf":
        s += f" w={p['w_ce']:g}"
    if p["fusion"] == "keep":
        s += f" m={p['m']}"
    return s


def main(model, cname, ks, n=None):
    vq, vl, _ = load_validation()
    ks = [int(x) for x in ks.split(",")]
    c = pd.read_parquet(CAND[cname])
    ids = set(vq["query_id"]) if n is None else sample_ids(vq["query_id"], int(n))
    c = c[c["query_id"].isin(ids)]
    lab = {q: vl[q] for q in ids}
    cache = pd.read_parquet(R.CACHE / f"ce_{model}_val.parquet")
    ce = c[c["rank"] <= max(ks)].merge(cache, on=["query_id", "item_id"], how="left")
    assert ce["ce"].notna().all(), f"не хватает CE-скоров: {ce['ce'].isna().sum()}"

    base = recall_at_k(R.top_lists(c), lab)
    rng = np.random.default_rng(0)
    qs = np.array(sorted(ids))
    perm = rng.permutation(len(qs))
    A, B = qs[perm[: len(qs) // 2]], qs[perm[len(qs) // 2:]]
    print(f"== {model} на {cname}, запросов {len(ids)}; A={len(A)}, B={len(B)}")
    print(f"исходный: все {base.mean():.4f}  A {base[A].mean():.4f}  B {base[B].mean():.4f}")
    ceil = {k: recall_at_k(R.top_lists(c, k), lab, k).mean() for k in ks}
    print("потолок Recall@K кандидатов:", {k: round(v, 4) for k, v in ceil.items()})

    rows, per = [], {}
    for p in grid(ks):
        out = R.fuse(c, ce, **p)
        r = recall_at_k(R.top_lists(out), lab)
        per[name(p)] = (r, out, p)
        d, lo, hi = bootstrap_diff(r, base)
        rows.append(dict(вариант=name(p), все=r.mean(), A=r[A].mean(), B=r[B].mean(), Δ_все=d, ДИ_lo=lo, ДИ_hi=hi))
    tab = pd.DataFrame(rows).set_index("вариант")
    print(tab.round(4).to_string())

    for sel_half, hold in (("A", B), ("B", A)):
        best = tab[sel_half].idxmax()
        r = per[best][0]
        d, lo, hi = bootstrap_diff(r[hold], base[hold])
        print(f"выбор на {sel_half}: {best}; на другой половине {r[hold].mean():.4f} против "
              f"{base[hold].mean():.4f}, Δ {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")
    best = tab["A"].idxmax()
    r, out, p = per[best]
    d, lo, hi = bootstrap_diff(r, base)
    print(f"\nлучший (по A): {best}; вся выборка {r.mean():.4f}, Δ {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")
    sl = pd.concat({"запросов": slices(base, vq)["запросов"], "исходный": slices(base, vq)["recall"],
                    "реранк": slices(r, vq)["recall"]}, axis=1)
    sl["Δ"] = sl["реранк"] - sl["исходный"]
    print(sl.round(4).to_string())

    # поднятые и выбитые правильные ответы
    pairs = pd.DataFrame([(q, i) for q, rel in lab.items() for i in rel], columns=["query_id", "item_id"])
    pairs = pairs.merge(c[["query_id", "item_id", "rank"]], how="left") \
        .merge(out[["query_id", "item_id", "rank"]].rename(columns={"rank": "new_rank"}), how="left")
    up = pairs[(pairs["rank"] > 50) & (pairs["new_rank"] <= 50)]
    down = pairs[(pairs["rank"] <= 50) & (pairs["new_rank"] > 50)]
    print(f"\nправильных пар: {len(pairs)}; поднято из 51–{p['k']} в топ-50: {len(up)}; выбито из топ-50: {len(down)}; "
          f"в 51–{p['k']} было: {int(((pairs['rank'] > 50) & (pairs['rank'] <= p['k'])).sum())}")
    top1 = out[out["rank"] == 1].set_index("query_id")["item_id"]  # что реранк поставил на 1-е место
    need = set(pd.concat([up, down])["item_id"]) | set(top1)
    items = R.load_items(need).set_index("item_id")
    qt = vq.set_index("query_id")
    ex = []
    for kind, df in (("поднят", up), ("выбит", down)):
        for _, x in df.sample(min(5, len(df)), random_state=0).iterrows():
            ex.append(dict(тип=kind, запрос=qt.loc[x["query_id"], "search_query"],
                           фильтр=qt.loc[x["query_id"], "search_infm_params_text"],
                           ответ=items.loc[x["item_id"], "item_title_raw"], было=int(x["rank"]),
                           стало=int(x["new_rank"]), top1=items.loc[top1[x["query_id"]], "item_title_raw"]))
    ex = pd.DataFrame(ex)
    ex.to_csv(f"work/rerank/examples_{model}_{cname}.csv", index=False)
    print(ex.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
