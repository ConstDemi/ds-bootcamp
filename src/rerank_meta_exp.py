"""Эксперименты: бонусы за фильтры и микрокатегорию поверх кандидатов гибрида (eda/11_meta_boost.md).

Запуск из корня репозитория:
    python src/rerank_meta_exp.py [--cands work/cands/hybrid_val_geo.parquet] [--bench work/cands/hybrid_bench_geo.parquet]
Классификатор для валидации учится на train[train_mask], для бенчмарка — на полном train.
Параметры подбираются на половине A валидации (seed 0), результат сообщается на половине B.
Пишет work/cands/hybrid_meta_{val,bench}.parquet (лучший вариант) и печатает таблицы для отчёта.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_meta as rm  # noqa: E402
from validation import add_flags, load_validation, normalize_text, recall_at_k, region_location_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, WORK = ROOT / "data", ROOT / "work"
K = 50


def recall_series(c: pd.DataFrame, labels: dict, k: int = K) -> pd.Series:
    """Recall@k по запросам из кандидатов (формат query_id, item_id, rank)."""
    top = c[c["rank"] <= k]
    preds = top.groupby("query_id")["item_id"].agg(list).to_dict()
    return recall_at_k(preds, labels, k)


def boot(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0):
    """Парный бутстреп разницы средних (a − b) по запросам: (разница, 2.5%, 97.5%)."""
    d = (a - b.reindex(a.index)).to_numpy()
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def slice_row(r: pd.Series, q: pd.DataFrame) -> dict:
    """Recall@50 по основным срезам валидации одной строкой."""
    d = q.set_index("query_id").loc[r.index]
    return {"все": r.mean(), "с фильтром": r[d.has_filter].mean(), "без фильтра": r[~d.has_filter].mean(),
            "текста нет в train": r[~d.text_in_train].mean(), "ответ без истории": r[~d.target_has_history].mean()}


def features(cands, queries, items, model, prf_top=20):
    """Признаки бонусов: совпадение «Вид/Тип услуги», P(микрокатегория), доля микрокатегории в топе (PRF)."""
    f = rm.filter_match(cands, queries, items)
    f = rm.microcat_prob(f, queries, items, model)
    f["prf"] = rm.prf_microcat(f, items, top=prf_top).to_numpy()
    return f


def best_apply(fam: str, par):
    """Функция cands_with_features -> переупорядоченные кандидаты для варианта из сетки."""
    if fam == "3 фильтр+P(mc)":
        a, ratio, b = par
        return lambda f: rm.rerank_bonus(f, w_vid=a, w_tip=a * ratio, w_mc=b)
    if fam == "3 фильтр+P(mc)+PRF":
        a, b, pw = par
        return lambda f: rm.rerank_bonus(f, w_vid=a, w_tip=a * 0.5, w_mc=b, w_prf=pw)
    if fam == "3 квота+P(mc)":
        n_res, b = par
        return lambda f: rm.rerank_quota(f, reserve=n_res, inner=rm.rerank_bonus(f, w_mc=b))
    if fam == "2 P(mc)":
        return lambda f: rm.rerank_bonus(f, w_mc=par)
    raise ValueError(fam)


def examples(old: pd.DataFrame, new: pd.DataFrame, labels: dict, vq: pd.DataFrame, items: pd.DataFrame,
             feats: pd.DataFrame, n: int = 8):
    """Печатает запросы, которые бонус вытащил в топ-50 и которые испортил."""
    titles = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_title_raw"]) \
        .set_index("item_id")["item_title_raw"]
    ro = old.set_index(["query_id", "item_id"])["rank"]
    rn = new.set_index(["query_id", "item_id"])["rank"]
    fx = feats.set_index(["query_id", "item_id"])[["m_vid", "m_tip", "p_mc"]]
    rows = []
    for q, s in labels.items():
        for i in s:
            a, b = ro.get((q, i), 999), rn.get((q, i), 999)
            if (a <= 50) != (b <= 50):
                fe = fx.loc[(q, i)] if (q, i) in fx.index else None
                rows.append((q, "вытащил" if b <= 50 else "испортил", a, b,
                             None if fe is None else round(float(fe.p_mc), 3),
                             None if fe is None else f"{int(fe.m_vid)}{int(fe.m_tip)}", str(titles.get(i, ""))[:60]))
    ex = pd.DataFrame(rows, columns=["query_id", "итог", "ранг до", "ранг после", "P(mc)", "Вид/Тип", "ответ"])
    ex = ex.merge(vq[["query_id", "search_query", "search_infm_params_text", "is_region"]], on="query_id")
    print(ex["итог"].value_counts().to_string())
    pd.set_option("display.max_colwidth", 70)
    for k in ("вытащил", "испортил"):
        print(f"--- {k}")
        print(ex[ex["итог"] == k].sample(min(n, (ex["итог"] == k).sum()), random_state=0).to_string(index=False))
    # Почему испортил: у ответа низкая P(mc) и/или нет совпадения по фильтру
    print(ex.groupby("итог")[["P(mc)"]].describe().round(3).to_string())
    ex.to_csv(WORK / "meta_examples.csv", index=False)


def main():
    """Подбор весов бонусов на половине A валидации и отчёт на B (eda/11_meta_boost.md)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(WORK / "cands" / "hybrid_val_geo.parquet"))
    ap.add_argument("--bench", default=str(WORK / "cands" / "hybrid_bench_geo.parquet"))
    ap.add_argument("--model-val", default=None, help="pickle готового классификатора на train[mask]")
    ap.add_argument("--model-bench", default=None, help="pickle готового классификатора на полном train")
    ap.add_argument("--out-suffix", default="hybrid_meta")
    args = ap.parse_args()
    t0 = time.time()

    vq, labels, mask = load_validation()
    items = rm.load_items(DATA / "benchmark_items.parquet")
    tr = pd.read_parquet(DATA / "train.parquet",
                         columns=["search_query", "search_infm_params_text", "item_microcat_id"])
    model = pickle.load(open(args.model_val, "rb")) if args.model_val else rm.fit_microcat(tr[mask])
    print(f"классификатор val готов, {time.time() - t0:.0f} с", flush=True)

    cands = pd.read_parquet(args.cands)
    f = features(cands, vq, items, model)
    base = recall_series(cands, labels)
    print(f"база Recall@50 = {base.mean():.4f}, @300 = {recall_series(cands, labels, 300).mean():.4f}")

    # Точность классификатора и доля ответов, совпадающих с фильтром
    mc = items.set_index("item_id")["item_microcat_id"]
    P = model.predict_proba(vq)
    top = model.classes_[np.argsort(-P, 1)[:, :5]]
    acc = {k: np.mean([bool({mc[i] for i in labels[q]} & set(top[j, :k]))
                       for j, q in enumerate(vq["query_id"])]) for k in (1, 3, 5)}
    print("точность микрокатегории top1/3/5:", {k: round(v, 3) for k, v in acc.items()})
    lab_pairs = pd.DataFrame([(q, i) for q, s in labels.items() for i in s], columns=["query_id", "item_id"])
    lf = rm.filter_match(lab_pairs, vq, items)
    hf = lf[lf.q_vid | lf.q_tip]
    print(f"ответы запросов со значимым фильтром: {len(hf)}, совпадают по Виду {hf[hf.q_vid].m_vid.mean():.3f}, "
          f"по Типу {hf[hf.q_tip].m_tip.mean():.3f}, по всем {hf.m_all.mean():.3f}")

    # Половины: A — подбор, B — отчёт
    qids = np.array(sorted(labels))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(qids))
    half_a, half_b = set(qids[perm[: len(qids) // 2]]), set(qids[perm[len(qids) // 2:]])
    ia, ib = base.index.isin(half_a), base.index.isin(half_b)

    variants = {}
    W = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 1.0]
    for w in W:
        variants[("1a Вид", w)] = dict(w_vid=w)
        variants[("1a Тип", w)] = dict(w_tip=w)
        variants[("1a Вид+Тип", w)] = dict(w_vid=w, w_tip=w)
        variants[("1a Вид+Тип/2", w)] = dict(w_vid=w, w_tip=w / 2)
    for w in [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03]:
        variants[("2 P(mc)", w)] = dict(w_mc=w)
        variants[("2 PRF-mc@20", w)] = dict(w_prf=w)
    for w in [0.001, 0.002, 0.003, 0.005, 0.01]:
        for kk in (1, 3):
            variants[(f"2 mc в top{kk}", w)] = dict(w_mc=w, mc_mode="topk", mc_topk=kk)

    rows, res = [], {}
    for (name, w), kw in variants.items():
        r = recall_series(rm.rerank_bonus(f, **kw), labels)
        res[(name, w)] = r
        rows.append((name, w, r[ia].mean(), r[ib].mean(), r.mean()))
    for n_res in (0, 5, 10, 20):
        r = recall_series(rm.rerank_quota(f, reserve=n_res), labels)
        res[("1b квота", n_res)] = r
        rows.append(("1b квота", n_res, r[ia].mean(), r[ib].mean(), r.mean()))
    grid = pd.DataFrame(rows, columns=["семейство", "параметр", "A", "B", "все"])
    grid.to_csv(WORK / "meta_grid.csv", index=False)

    # Лучшие параметры каждого семейства по половине A
    best = grid.loc[grid.groupby("семейство")["A"].idxmax()]
    print(best.round(4).to_string(index=False), flush=True)

    # Комбинации: лучшие фильтр-бонус × mc-бонус, подбор на A
    fw = [0.002, 0.003, 0.005, 0.01, 0.02]
    mw = [0.002, 0.003, 0.005, 0.007, 0.01]
    comb = []
    for a in fw:
        for b in mw:
            for tip_ratio in (0.5, 1.0):
                kw = dict(w_vid=a, w_tip=a * tip_ratio, w_mc=b)
                r = recall_series(rm.rerank_bonus(f, **kw), labels)
                res[("3 фильтр+P(mc)", (a, tip_ratio, b))] = r
                comb.append(("3 фильтр+P(mc)", (a, tip_ratio, b), r[ia].mean(), r[ib].mean(), r.mean()))
            for pw in (0.001, 0.002):
                kw = dict(w_vid=a, w_tip=a * 0.5, w_mc=b, w_prf=pw)
                r = recall_series(rm.rerank_bonus(f, **kw), labels)
                res[("3 фильтр+P(mc)+PRF", (a, b, pw))] = r
                comb.append(("3 фильтр+P(mc)+PRF", (a, b, pw), r[ia].mean(), r[ib].mean(), r.mean()))
            for n_res in (0, 5, 10):
                inner = rm.rerank_bonus(f, w_mc=b)
                r = recall_series(rm.rerank_quota(f, reserve=n_res, inner=inner), labels)
                res[("3 квота+P(mc)", (n_res, b))] = r
                comb.append(("3 квота+P(mc)", (n_res, b), r[ia].mean(), r[ib].mean(), r.mean()))
    comb = pd.DataFrame(comb, columns=["семейство", "параметр", "A", "B", "все"]).drop_duplicates(["семейство", "параметр"])
    grid = pd.concat([grid, comb]).reset_index(drop=True)
    grid.to_csv(WORK / "meta_grid.csv", index=False)
    best = grid.loc[grid.groupby("семейство")["A"].idxmax()].sort_values("A", ascending=False)

    # Отчёт: лучшая точка семейства (по A) → результат на B, Δ с интервалом, срезы
    vqi = vq.set_index("query_id")
    out = []
    for _, row in best.iterrows():
        r = res[(row["семейство"], row["параметр"])]
        dB = boot(r[ib], base[ib])
        dAll = boot(r, base)
        sl = slice_row(r[ib], vq)
        out.append({"вариант": row["семейство"], "параметр": row["параметр"], "A": row["A"], "B": row["B"],
                    "ΔB": dB[0], "ΔB 95%": f"[{dB[1]:+.4f}; {dB[2]:+.4f}]", "все": r.mean(),
                    "Δвсе": dAll[0], "Δвсе 95%": f"[{dAll[1]:+.4f}; {dAll[2]:+.4f}]",
                    **{f"B {k}": v for k, v in sl.items()}})
    rep = pd.DataFrame(out)
    base_sl = slice_row(base[ib], vq)
    print("база на B:", {k: round(v, 4) for k, v in base_sl.items()}, "база на всех:", round(base.mean(), 4))
    print("база на всех по срезам:", {k: round(v, 4) for k, v in slice_row(base, vq).items()})
    pd.set_option("display.width", 250)
    print(rep.round(4).to_string(index=False))
    rep.to_csv(WORK / "meta_report.csv", index=False)

    # Лучший вариант (по A) → срезы на всех 3000 и примеры
    top_row = best.iloc[0]
    print("лучший по A:", top_row["семейство"], top_row["параметр"])
    rbest = res[(top_row["семейство"], top_row["параметр"])]
    print("лучший на всех по срезам:", {k: round(v, 4) for k, v in slice_row(rbest, vq).items()})
    pickle.dump({"best": (top_row["семейство"], top_row["параметр"]), "res": res, "base": base,
                 "half_a": half_a, "half_b": half_b}, open(WORK / "meta_res.pkl", "wb"))

    # Параметры лучшего варианта -> функция переупорядочивания
    fam, par = top_row["семейство"], top_row["параметр"]
    apply = best_apply(fam, par)
    new_val = apply(f)
    assert abs(recall_series(new_val, labels).mean() - rbest.mean()) < 1e-9
    new_val.to_parquet(WORK / "cands" / f"{args.out_suffix}_val.parquet", index=False)
    examples(cands, new_val, labels, vq, items, f)

    # Бенчмарк: классификатор на полном train, флаги — не нужны
    del f, res
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    model_b = pickle.load(open(args.model_bench, "rb")) if args.model_bench else rm.fit_microcat(tr)
    cb = pd.read_parquet(args.bench)
    fb = features(cb, bq, items, model_b)
    nb = apply(fb)
    assert nb.groupby("query_id").size().eq(cb.groupby("query_id").size()).all()
    nb.to_parquet(WORK / "cands" / f"{args.out_suffix}_bench.parquet", index=False)
    top50_same = (nb[nb["rank"] <= 50].merge(cb[cb["rank"] <= 50], on=["query_id", "item_id"]).shape[0]
                  / cb[cb["rank"] <= 50].shape[0])
    print(f"бенчмарк: переупорядочено {nb.query_id.nunique()} запросов, топ-50 совпадает с исходным на {top50_same:.3f}")
    print(f"готово, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
