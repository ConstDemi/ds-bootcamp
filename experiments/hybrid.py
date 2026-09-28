"""Гибрид sparse (BM25) + dense (эмбеддинги, см. eda/07_dense.md): слияние готовых кандидатов топ-300.

Ретриверы здесь не считаются: читаем work/cands/{sparse,dense}_{val,bench}_{all,geo}.parquet
(query_id, item_id, score, rank) и сливаем их одной и той же функцией для валидации и бенчмарка.

Особенность geo: для городского запроса список источника — сначала объявления города, потом добор
из всего корпуса со своим (глобальным) скором, поэтому score может расти по rank. Чтобы не смешивать
несравнимые скоры, делим кандидатов на два яруса: «город» (item_location_id == search_location_id)
и «добор». Нормировка и ранги считаются внутри (запрос, источник, ярус), а в выдаче ярус «город»
всегда стоит выше «добора» (как у самих источников). У регионов и «всей России» совпадений локации
нет, поэтому весь список — один ярус «добор», то есть обычное слияние.
Для RRF ярусы можно выключить (tiers=False): тогда берутся исходные ранги geo-списков, в которых
город уже стоит первым, — «мягкое» гео. Этот вариант и выбран (см. CHOSEN).

Методы:
    rrf        — score = Σ_источники w_s / (k_rrf + rank_s), rank_s — ранг внутри яруса;
    weighted   — (1 - w) * norm(sparse) + w * norm(dense), norm — min-max или z-score
                 по списку источника внутри (запрос, ярус); отсутствующий в списке — минимум списка;
    interleave — sparse1, dense1, sparse2, dense2, ... без повторов (контроль).

Запуск из корня репозитория:  python experiments/hybrid.py   (~1.5 мин, ~3 ГБ)
Пишет answer_02_hybrid.csv и work/cands/hybrid_{val,bench}_geo.parquet.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baselines import write_answer  # noqa: E402
from validation import load_validation, recall_at_k, region_location_ids, slices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CANDS = ROOT / "work" / "cands"
SOURCES = ("sparse", "dense")
N_OUT = 300          # сколько кандидатов гибрида сохраняем
KS = (50, 100, 200, 300)
# Выбранный метод (см. eda/08_hybrid.md): все гибриды в пределах шума друг от друга, RRF не зависит
# от шкалы скоров и почти не зависит от k, поэтому переживёт замену dense на GPU-версию без перенастройки.
# Ярусы для RRF не нужны: ранги geo-списков уже «сначала город», а явный ярус поднимает над релевантным
# добором sparse сотни нерелевантных городских объявлений dense (у 23% городских запросов у sparse
# меньше 50 городских кандидатов). Для weighted ярусы обязательны (скоры города и добора несравнимы).
CHOSEN = "rrf k=60, без ярусов"


# ---------------------------------------------------------------- загрузка

def item_locations() -> pd.Series:
    """item_id -> item_location_id по корпусу."""
    it = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_location_id"])
    return it.set_index(it["item_id"].astype(str))["item_location_id"]


def load_cands(split: str, mode: str, queries: pd.DataFrame, item_loc: pd.Series) -> dict:
    """{источник: DataFrame(query_id, item_id, score, rank, city)} для split ∈ {val, bench}.

    city — объявление из города запроса (ярус «город»); для регионов всегда False.
    """
    qloc = queries.set_index("query_id")["search_location_id"]
    out = {}
    for s in SOURCES:
        c = pd.read_parquet(CANDS / f"{s}_{split}_{mode}.parquet")
        c["query_id"] = c["query_id"].astype(str)
        c["item_id"] = c["item_id"].astype(str)
        c["city"] = (c["item_id"].map(item_loc).to_numpy()
                     == c["query_id"].map(qloc).to_numpy())
        out[s] = c.sort_values(["query_id", "rank"], kind="stable").reset_index(drop=True)
    return out


# ---------------------------------------------------------------- слияние

def _group_keys(c: pd.DataFrame, tiers: bool) -> list:
    return ["query_id", "city"] if tiers else ["query_id"]


def _tier_rank(c: pd.DataFrame, tiers: bool) -> pd.Series:
    """Ранг внутри (запрос[, ярус]) в исходном порядке источника, с 1."""
    return c.groupby(_group_keys(c, tiers), sort=False).cumcount() + 1


def _normalize(c: pd.DataFrame, how: str, tiers: bool) -> pd.Series:
    """Скор источника, нормированный внутри (запрос[, ярус])."""
    g = c.groupby(_group_keys(c, tiers), sort=False)["score"]
    s = c["score"].astype("float64")
    if how == "minmax":
        lo, hi = g.transform("min"), g.transform("max")
        rng = (hi - lo).to_numpy()
        # один кандидат или равные скоры: считаем его лучшим в своём списке
        return pd.Series(np.where(rng > 0, (s - lo) / np.where(rng > 0, rng, 1), 1.0), index=c.index)
    if how == "z":
        mu, sd = g.transform("mean"), g.transform("std").fillna(0)
        return pd.Series(np.where(sd > 0, (s - mu) / sd.where(sd > 0, 1), 0.0), index=c.index)
    raise ValueError(how)


def _finish(parts: pd.DataFrame, tiers: bool) -> pd.DataFrame:
    """Сортировка слитых кандидатов: ярус «город» выше, внутри — по score. Возвращает с rank."""
    by = ["query_id", "city", "score"] if tiers else ["query_id", "score"]
    asc = [True, False, False] if tiers else [True, False]
    f = parts.sort_values(by, ascending=asc, kind="stable")
    f["rank"] = f.groupby("query_id", sort=False).cumcount() + 1
    return f[["query_id", "item_id", "score", "rank", "city"]].reset_index(drop=True)


def fuse_rrf(cands: dict, k_rrf: int = 60, weights: dict | None = None, tiers: bool = True) -> pd.DataFrame:
    """Reciprocal Rank Fusion: Σ w_s / (k_rrf + rank_s); отсутствие в списке даёт 0."""
    parts = []
    for s, c in cands.items():
        w = 1.0 if weights is None else weights[s]
        parts.append(pd.DataFrame({"query_id": c["query_id"], "item_id": c["item_id"], "city": c["city"],
                                   "score": w / (k_rrf + _tier_rank(c, tiers))}))
    p = pd.concat(parts).groupby(["query_id", "item_id"], sort=False) \
        .agg(score=("score", "sum"), city=("city", "first")).reset_index()
    return _finish(p, tiers)


def fuse_weighted(cands: dict, w_dense: float = 0.5, how: str = "minmax", tiers: bool = True) -> pd.DataFrame:
    """(1 - w) * norm(sparse) + w * norm(dense). Нет в списке источника — берём минимум его списка
    (для min-max это 0) в том же (запрос, ярус); если у источника такого яруса нет вовсе — минимум по запросу."""
    ws = {"sparse": 1.0 - w_dense, "dense": w_dense}
    wide = None
    floors = {}
    for s, c in cands.items():
        n = _normalize(c, how, tiers)
        d = pd.DataFrame({"query_id": c["query_id"], "item_id": c["item_id"], "city": c["city"], s: n})
        floors[s] = (d.groupby(_group_keys(d, tiers))[s].min(), d.groupby("query_id")[s].min())
        d = d.drop_duplicates(["query_id", "item_id"])
        wide = d if wide is None else wide.merge(d, on=["query_id", "item_id"], how="outer",
                                                  suffixes=("", "_r"))
        if "city_r" in wide:
            wide["city"] = wide["city"].fillna(wide.pop("city_r"))
    wide["city"] = wide["city"].astype(bool)
    score = np.zeros(len(wide))
    for s in cands:
        v = wide[s]
        if v.isna().any():
            by_tier, by_q = floors[s]
            key = pd.MultiIndex.from_frame(wide[_group_keys(wide, tiers)]) if tiers else wide["query_id"]
            fill = pd.Series(by_tier.reindex(key).to_numpy(), index=wide.index)
            fill = fill.fillna(wide["query_id"].map(by_q)).fillna(0.0)
            v = v.fillna(fill)
        score += ws[s] * v.to_numpy()
    wide["score"] = score
    return _finish(wide, tiers)


def fuse_interleave(cands: dict, first: str = "sparse", tiers: bool = True) -> pd.DataFrame:
    """Чередование: ранг внутри (запрос[, ярус]) r даёт позицию 2r-1 первому источнику и 2r второму;
    повторы убираем, оставляя лучшую позицию."""
    parts = []
    for s, c in cands.items():
        r = _tier_rank(c, tiers).to_numpy()
        pos = 2 * r - (1 if s == first else 0)
        parts.append(pd.DataFrame({"query_id": c["query_id"], "item_id": c["item_id"], "city": c["city"],
                                   "score": -pos.astype(float)}))
    p = pd.concat(parts).sort_values("score", ascending=False, kind="stable") \
        .drop_duplicates(["query_id", "item_id"])
    return _finish(p, tiers)


# ---------------------------------------------------------------- выдача

def top_n(fused: pd.DataFrame, query_ids, fallbacks: list[pd.DataFrame], n: int = N_OUT) -> pd.DataFrame:
    """Ровно n уникальных id на каждый запрос (если хватает источников): топ-n слияния,
    затем добор по порядку из fallbacks (например, списки режима all), без повторов."""
    parts = [fused[["query_id", "item_id", "score"]].assign(_src=0, _r=fused["rank"])]
    for i, fb in enumerate(fallbacks, 1):
        fb = fb.sort_values(["query_id", "rank"])
        parts.append(pd.DataFrame({"query_id": fb["query_id"].astype(str), "item_id": fb["item_id"].astype(str),
                                   "score": 0.0, "_src": i, "_r": fb["rank"]}))
    p = pd.concat(parts, ignore_index=True)
    p = p[p["query_id"].isin(set(query_ids))]
    p = p.sort_values(["query_id", "_src", "_r"], kind="stable").drop_duplicates(["query_id", "item_id"])
    p["rank"] = p.groupby("query_id", sort=False).cumcount() + 1
    p = p[p["rank"] <= n].astype({"score": "float32", "rank": "int32"})  # типы как у sparse/dense
    return p[["query_id", "item_id", "score", "rank"]].reset_index(drop=True)


def to_preds(df: pd.DataFrame, k: int) -> dict:
    """query_id -> список из k первых item_id (df уже отсортирован по rank внутри запроса)."""
    d = df[df["rank"] <= k]
    return {q: g.tolist() for q, g in d.groupby("query_id", sort=False)["item_id"]}


# ---------------------------------------------------------------- замеры

def recall_curve(df: pd.DataFrame, labels: dict, ks=KS) -> dict:
    return {k: recall_at_k(to_preds(df, k), labels, k).mean() for k in ks}


def bootstrap_diff(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0):
    """Парный бутстреп по запросам: среднее (a - b) и 95% интервал."""
    d = (a - b.loc[a.index]).to_numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def overlap(src: dict, labels: dict, k: int) -> pd.Series:
    """Сколько пар (запрос, правильный id) нашёл в топ-k только sparse / только dense / оба / никто."""
    tops = {s: set(zip(c.loc[c["rank"] <= k, "query_id"], c.loc[c["rank"] <= k, "item_id"]))
            for s, c in src.items()}
    pairs = [(q, i) for q, rel in labels.items() for i in rel]
    sp = np.array([p in tops["sparse"] for p in pairs])
    de = np.array([p in tops["dense"] for p in pairs])
    return pd.Series({"только sparse": int((sp & ~de).sum()), "только dense": int((~sp & de).sum()),
                      "оба": int((sp & de).sum()), "никто": int((~sp & ~de).sum()), "всего": len(pairs)})


METHODS = {  # имя -> функция(cands, tiers); сетка из задания + контрольные варианты
    **{f"rrf k={k}": (lambda c, t, k=k: fuse_rrf(c, k, tiers=t)) for k in (10, 30, 60)},
    **{f"weighted minmax w={w}": (lambda c, t, w=w: fuse_weighted(c, w, "minmax", t)) for w in (0.3, 0.5, 0.7)},
    **{f"weighted z w={w}": (lambda c, t, w=w: fuse_weighted(c, w, "z", t)) for w in (0.3, 0.5, 0.7)},
    "interleave": lambda c, t: fuse_interleave(c, tiers=t),
    # абляции: без ярусов (нормировка/ранги по всему списку geo вперемешку)
    "rrf k=60, без ярусов": lambda c, t: fuse_rrf(c, 60, tiers=False),
    # проверка k после замены dense на GPU-версию (eda/08_hybrid.md, «Гибрид с GPU-dense»)
    "rrf k=30, без ярусов": lambda c, t: fuse_rrf(c, 30, tiers=False),
    "weighted minmax w=0.5, без ярусов": lambda c, t: fuse_weighted(c, 0.5, "minmax", tiers=False),
}


def main():
    t0 = time.time()
    vq, vl, _ = load_validation()
    vq["query_id"] = vq["query_id"].astype(str)
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    bq["query_id"] = bq["query_id"].astype(str)
    iloc = item_locations()

    rows = []
    cands_val = {mode: load_cands("val", mode, vq, iloc) for mode in ("all", "geo")}
    for mode in ("geo", "all"):
        c = cands_val[mode]
        # добор: для geo — списки режима all обоих источников; в all добор не нужен (dense даёт 300)
        fb = [cands_val["all"][s] for s in SOURCES] if mode == "geo" else []
        for s in SOURCES:
            r = recall_at_k(to_preds(c[s], 50), vl).mean()
            rows.append({"метод": f"{s}-only", "режим": mode, "R@50": r})
        for name, fn in METHODS.items():
            if mode == "all" and "без ярусов" in name:
                continue  # в all ярусов нет, варианты совпадают
            h = top_n(fn(c, mode == "geo"), vq["query_id"], fb)  # в all ярусов нет
            rows.append({"метод": name, "режим": mode, "R@50": recall_at_k(to_preds(h, 50), vl).mean()})
        print(f"{mode}: готово за {time.time() - t0:.0f} с", flush=True)
    tab = pd.DataFrame(rows).pivot(index="метод", columns="режим", values="R@50")[["geo", "all"]]
    tab = tab.sort_values("geo", ascending=False)
    print("\nRecall@50 на валидации\n", tab.round(4).to_string(), sep="")

    # ---- лучший (выбор по geo; см. отчёт про шум) и контроль переобучения: 2 половины валидации
    cg, fb_all = cands_val["geo"], [cands_val["all"][s] for s in SOURCES]
    grid = [n for n in METHODS if "без ярусов" not in n or n == CHOSEN]
    per_q = {n: recall_at_k(to_preds(top_n(METHODS[n](cg, True), vq["query_id"], fb_all), 50), vl) for n in grid}
    rng = np.random.default_rng(0)
    half = set(rng.choice(vq["query_id"].to_numpy(), len(vq) // 2, replace=False))
    for part, other in (("A", "B"), ("B", "A")):
        m_in = {n: r[r.index.isin(half) == (part == "A")].mean() for n, r in per_q.items()}
        m_out = {n: r[r.index.isin(half) != (part == "A")].mean() for n, r in per_q.items()}
        best_in = max(m_in, key=m_in.get)
        print(f"выбор на половине {part}: {best_in} ({m_in[best_in]:.4f}), на {other}: {m_out[best_in]:.4f}")

    best = max(per_q, key=lambda n: per_q[n].mean())
    print(f"\nлучший по geo: {best} ({per_q[best].mean():.4f}); выбран: {CHOSEN} ({per_q[CHOSEN].mean():.4f})")
    src_r = {s: recall_at_k(to_preds(cg[s], 50), vl) for s in SOURCES}
    for a, b in ((best, CHOSEN), (CHOSEN, "rrf k=60"), (CHOSEN, "sparse"), (CHOSEN, "dense"), (CHOSEN, "interleave")):
        ra = per_q[a] if a in per_q else src_r[a]
        rb = per_q[b] if b in per_q else src_r[b]
        d, lo, hi = bootstrap_diff(ra, rb)
        print(f"бутстреп {a} − {b}: {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")

    # ---- кривая и срезы выбранного гибрида
    hv = top_n(METHODS[CHOSEN](cg, True), vq["query_id"], fb_all)
    curves = pd.DataFrame({"sparse": recall_curve(cg["sparse"], vl), "dense": recall_curve(cg["dense"], vl),
                           "гибрид": recall_curve(hv, vl)}).T
    # потолок: объединение обоих топ-300 (до 600 id) — сколько вообще есть среди кандидатов
    union = pd.concat([cg[s][["query_id", "item_id"]] for s in SOURCES]).drop_duplicates()
    u = union.groupby("query_id")["item_id"].agg(set)
    ceil = np.mean([len(rel & u.get(q, set())) / len(rel) for q, rel in vl.items()])
    print("\nкривая Recall@k (geo)\n", curves.round(4).to_string(), f"\nобъединение топ-300 обоих: {ceil:.4f}", sep="")
    sl = pd.concat({"sparse": slices(src_r["sparse"], vq)["recall"], "dense": slices(src_r["dense"], vq)["recall"],
                    "гибрид": slices(per_q[CHOSEN], vq)["recall"],
                    "rrf k=60 (ярусы)": slices(per_q["rrf k=60"], vq)["recall"],
                    best: slices(per_q[best], vq)["recall"]}, axis=1)
    sl.insert(0, "запросов", slices(per_q[CHOSEN], vq)["запросов"].astype(int))
    print("\nсрезы Recall@50 (geo)\n", sl.round(3).to_string(), sep="")

    # ---- взаимодополняемость источников
    ov = pd.DataFrame({f"топ-{k}": overlap(cg, vl, k) for k in (50, 300)})
    hit50 = set(zip(hv.loc[hv["rank"] <= 50, "query_id"], hv.loc[hv["rank"] <= 50, "item_id"]))
    ov.loc["гибрид@50 нашёл", "топ-50"] = sum((q, i) in hit50 for q, rel in vl.items() for i in rel)
    print("\nпары (запрос, правильный id), geo\n", ov.to_string(), sep="")

    # ---- промахи выбранного гибрида (правильного нет в топ-50), случайные 8
    titles = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_title_raw"])
    titles = titles.set_index(titles["item_id"].astype(str))["item_title_raw"].str.slice(0, 60)
    miss = per_q[CHOSEN][per_q[CHOSEN] == 0].index
    rank_of = {s: cg[s].set_index(["query_id", "item_id"])["rank"] for s in SOURCES}
    hrank = hv.set_index(["query_id", "item_id"])["rank"]
    qi = vq.set_index("query_id")
    print(f"\nпромахов: {len(miss)} из {len(vq)}")
    for q in pd.Series(miss).sample(8, random_state=1):
        rel = sorted(vl[q])[0]
        top3 = "; ".join(titles[hv.loc[(hv["query_id"] == q) & (hv["rank"] <= 3), "item_id"]])
        rk = {s: rank_of[s].get((q, rel), "—") for s in SOURCES}
        print(f"- «{qi.at[q, 'search_query']}» [{qi.at[q, 'search_infm_params_text'] or ''}] "
              f"регион={qi.at[q, 'is_region']} | ответ: {titles[rel]} | "
              f"ранг sparse/dense/гибрид: {rk['sparse']}/{rk['dense']}/{hrank.get((q, rel), '—')} | топ-3: {top3}")

    # ---- бенчмарк: тот же метод, добор из all обоих источников
    cb = load_cands("bench", "geo", bq, iloc)
    cba = load_cands("bench", "all", bq, iloc)
    hb = top_n(METHODS[CHOSEN](cb, True), bq["query_id"], [cba[s] for s in SOURCES])
    n_per = hb.groupby("query_id").size().reindex(bq["query_id"]).fillna(0)
    assert (n_per >= 50).all(), "у части запросов бенчмарка меньше 50 кандидатов"
    write_answer(to_preds(hb, 50), ROOT / "answer_02_hybrid.csv", bq)
    hv.to_parquet(CANDS / "hybrid_val_geo.parquet", index=False)
    hb.to_parquet(CANDS / "hybrid_bench_geo.parquet", index=False)
    breg = bq["search_location_id"].isin(region_location_ids())
    print(f"\nбенчмарк: {len(bq)} запросов, кандидатов на запрос min {int(n_per.min())}, регионов {breg.mean():.1%}; "
          f"записано answer_02_hybrid.csv, work/cands/hybrid_{{val,bench}}_geo.parquet")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
