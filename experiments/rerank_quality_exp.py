"""Эксперименты с бонусами за качество объявления (eda/15_quality.md) поверх hybrid_geo2_meta_{val,bench}.

1. Разведка: признаки правильных ответов против остальных кандидатов того же запроса (и соседей по рангу ±10).
2. Сетка весов для каждого признака отдельно и покоординатный подъём для комбинации — подбор на половине A
   (seed 0, как в eda/11), результат на половине B, парный бутстреп Δ по запросам.
3. Срезы для выбранного варианта на всех 3000, сколько правильных ответов поднято в топ-50 / выбито.
4. Сдвиг признаков: топ-50 валидации против топ-50 бенчмарка.
5. Сохраняет work/cands/hybrid_geo2_meta_q_{val,bench}.parquet, сетку work/quality_grid.csv.
Запуск из корня репозитория: python experiments/rerank_quality_exp.py (~1 мин, < 2 ГБ, только CPU).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rerank_quality as rq  # noqa: E402
from validation import load_validation, slices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, CANDS, WORK = ROOT / "data", ROOT / "work" / "cands", ROOT / "work"
K, N = 50, 300

# признак -> знак бонуса (для бинарных «плохих» признаков вес отрицательный)
FEATS = {"pct_rev_log": +1, "pct_desc_log": +1, "pct_rating_f": +1, "rating_c": +1, "has_rev": +1,
         "has_rating": +1, "pct_title_len": +1, "phone_hidden": -1, "msg_forbidden": -1, "price_stub": -1,
         "pct_price_log": -1, "rev_log": +1, "desc_log": +1}
GRID = [1e-4, 2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2]


def boot(d: np.ndarray, n: int = 2000, seed: int = 0):
    """Парный бутстреп средней разницы по запросам: (Δ, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


class Evaluator:
    """Быстрый Recall@K: кандидаты лежат матрицей (запросы × 300), пересортировка — через порог k-го скора."""

    def __init__(self, c: pd.DataFrame, labels: dict):
        self.qids = c["query_id"].to_numpy()[::N]
        self.nq = len(self.qids)
        # тай-брейк по старому рангу: крошечная добавка, чтобы равные скоры не меняли порядок
        self.S = (c["score"].to_numpy(np.float64) - c["rank"].to_numpy() * 1e-12).reshape(self.nq, N)
        self.F = {k: c[k].to_numpy(np.float64).reshape(self.nq, N) for k in FEATS}
        lab = pd.DataFrame([(q, i) for q, s in labels.items() for i in s], columns=["query_id", "item_id"])
        lab["y"] = True
        y = c[["query_id", "item_id"]].merge(lab, on=["query_id", "item_id"], how="left")["y"]
        self.Y = y.fillna(False).to_numpy(bool).reshape(self.nq, N)
        self.nlab = np.array([len(labels[q]) for q in self.qids])

    def recall(self, w: dict, k: int = K) -> pd.Series:
        s = self.S.copy()
        for f, v in w.items():
            if v:
                s += v * self.F[f]
        thr = -np.partition(-s, k - 1, axis=1)[:, k - 1:k]
        hits = (self.Y & (s >= thr)).sum(1)
        return pd.Series(hits / self.nlab, index=self.qids)


def fit_clogit(ev: "Evaluator", mask: np.ndarray, fs: list) -> dict:
    """Условный логит по запросам: P(объявление | запрос) ∝ exp(b·score + Σ g_f·f) по 300 кандидатам.

    Гладкая цель по всем позициям правильных ответов шумит меньше, чем Recall@50 на сетке.
    Вес бонуса в единицах score: w_f = g_f / b. Используются только запросы из mask с ответом в топ-300.
    """
    from scipy.optimize import minimize
    m = mask & ev.Y.any(1)
    X = np.stack([ev.S[m] * 100] + [ev.F[f][m] for f in fs], -1)  # score×100 — для обусловленности
    Y = ev.Y[m].astype(float)
    Y /= Y.sum(1, keepdims=True)

    def nll(t):
        u = X @ t
        u -= u.max(1, keepdims=True)
        p = np.exp(u)
        p /= p.sum(1, keepdims=True)
        return -(Y * np.log(p + 1e-300)).sum(), -((Y - p)[..., None] * X).sum((0, 1))

    t = minimize(nll, np.r_[1.0, np.zeros(len(fs))], jac=True, method="L-BFGS-B").x
    return {f: float(t[i + 1] / (t[0] * 100)) for i, f in enumerate(fs)}


def explore(c: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Таблица: правильные ответы vs все остальные кандидаты тех же запросов vs соседи по рангу ±10."""
    lab = pd.DataFrame([(q, i) for q, s in labels.items() for i in s], columns=["query_id", "item_id"])
    lab["y"] = 1
    c = c.merge(lab, on=["query_id", "item_id"], how="left")
    c["y"] = c["y"].fillna(0)
    c = c[c["query_id"].isin(c.loc[c.y == 1, "query_id"])]
    cols = ["rev_log", "has_rev", "has_rating", "rating_c", "price_stub", "phone_hidden", "msg_forbidden",
            "desc_log", "title_len", "pct_rev_log", "pct_desc_log", "pct_rating_f", "pct_title_len", "pct_price_log"]
    pos = c[c.y == 1]
    nb = c[c.y == 0].merge(pos[["query_id", "rank"]].rename(columns={"rank": "r0"}), on="query_id")
    nb = nb[(nb["rank"] - nb["r0"]).abs() <= 10]
    t = pd.DataFrame({"правильные": pos[cols].mean(), "остальные": c.loc[c.y == 0, cols].mean(),
                      "соседи ±10": nb[cols].mean()})
    # доля правильных, у которых признак выше медианы своего запроса (для pct_*) — 0.5 = нет эффекта
    print(f"правильных в топ-300: {len(pos)}; соседей: {len(nb)}")
    return t


def main():
    t0 = time.time()
    vq, labels, _ = load_validation()
    items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=rq.QUALITY_COLS)
    qf = rq.item_quality_features(items)
    del items
    cv = pd.read_parquet(CANDS / "hybrid_geo2_meta_val.parquet")
    fv = rq.candidate_features(cv, qf)

    print("=== 1. Разведка ===")
    print(explore(fv, labels).round(3).to_string())
    dq_all = vq.set_index("query_id")
    for h, name in ((True, "у ответа есть история"), (False, "ответ без истории")):
        qs = set(dq_all.index[dq_all["target_has_history"] == h])
        print(f"--- {name} ---")
        print(explore(fv[fv["query_id"].isin(qs)], {q: labels[q] for q in qs}).round(3).to_string())

    ev = Evaluator(fv, labels)
    qids = np.array(sorted(labels))
    perm = np.random.default_rng(0).permutation(len(qids))
    half_b = set(qids[perm[len(qids) // 2:]])
    ib = np.array([q in half_b for q in ev.qids])
    ia = ~ib
    # ответ без истории: на бенчмарке таких почти все (eda/05: популярность там ~0.01 против 0.146 на валидации)
    nh = ~vq.set_index("query_id").loc[ev.qids, "target_has_history"].to_numpy()
    base = ev.recall({})
    print(f"\nбаза: все {base.mean():.4f}, A {base[ia].mean():.4f}, B {base[ib].mean():.4f}, "
          f"@300 {ev.recall({}, 300).mean():.4f}")

    # === 2a. Одиночные признаки ===
    rows = []
    for f, sg in FEATS.items():
        for w in GRID:
            r = ev.recall({f: sg * w})
            rows.append(("single", f, sg * w, r[ia].mean(), r[ib].mean(), r.mean()))
    grid = pd.DataFrame(rows, columns=["семейство", "признак", "w", "A", "B", "все"])

    print("\n=== 2a. Лучший вес каждого признака по A ===")
    single_rows = []
    for f in FEATS:
        g = grid[grid["признак"] == f]
        best = g.loc[g["A"].idxmax()]
        r = ev.recall({f: best.w})
        d = boot((r - base)[ib].to_numpy())
        single_rows.append((f, best.w, best.A, best.B, *d))
    singles = pd.DataFrame(single_rows, columns=["признак", "w", "A", "B", "dB", "lo", "hi"])
    print(singles.round(4).to_string(index=False))

    # === 2b. Комбинация: покоординатный подъём по A из нуля, 3 прохода ===
    cand = ["pct_rev_log", "pct_desc_log", "rating_c", "pct_rating_f", "msg_forbidden", "phone_hidden",
            "price_stub", "pct_title_len", "pct_price_log", "has_rev"]
    w = {f: 0.0 for f in cand}
    for it in range(3):
        for f in cand:
            best_v, best_a = w[f], ev.recall(w)[ia].mean()
            for v in [0.0] + [FEATS[f] * g for g in GRID]:
                a = ev.recall({**w, f: v})[ia].mean()
                if a > best_a + 1e-9:
                    best_v, best_a = v, a
            w[f] = best_v
        r = ev.recall(w)
        print(f"проход {it}: A {r[ia].mean():.4f}, B {r[ib].mean():.4f}, веса {{{', '.join(f'{k}: {v:g}' for k, v in w.items() if v)}}}")
    w_cd = {k: v for k, v in w.items() if v}

    # 2c. Явная сетка по двум сильнейшим признакам (отзывы × описание) + рейтинг
    rows2 = []
    for a in [0, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2]:
        for b in [0, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3]:
            for rc in [0, 1e-3, 2e-3, 4e-3]:
                ww = {"pct_rev_log": a, "pct_desc_log": b, "rating_c": rc}
                r = ev.recall(ww)
                rows2.append(("rev+desc+rating", f"{a:g}/{b:g}/{rc:g}", 0, r[ia].mean(), r[ib].mean(), r.mean()))
    g2 = pd.DataFrame(rows2, columns=grid.columns)
    grid = pd.concat([grid, g2], ignore_index=True)
    grid.to_csv(WORK / "quality_grid.csv", index=False)
    best2 = g2.loc[g2["A"].idxmax()]
    a_, b_, rc_ = map(float, best2["признак"].split("/"))
    w_grid = {k: v for k, v in {"pct_rev_log": a_, "pct_desc_log": b_, "rating_c": rc_}.items() if v}

    # масштаб лучшего набора — проверка устойчивости (только показываем, не выбираем по B)
    print("\n=== 2d. Варианты (веса по A; B — честная оценка) ===")
    w_lA = fit_clogit(ev, ia, ["pct_rev_log", "pct_desc_log"])
    w_lAn = fit_clogit(ev, ia & nh, ["pct_rev_log", "pct_desc_log"])
    w_lAn3 = fit_clogit(ev, ia & nh, ["pct_rev_log", "pct_desc_log", "rating_c"])
    w_lAall = fit_clogit(ev, ia & nh, [f for f in FEATS if f not in ("rev_log", "desc_log", "has_rev", "has_rating")])
    variants = {
        "отзывы (pct)": {"pct_rev_log": singles.set_index("признак").loc["pct_rev_log", "w"]},
        "описание (pct)": {"pct_desc_log": singles.set_index("признак").loc["pct_desc_log", "w"]},
        "сетка отзывы+описание+рейтинг": w_grid,
        "покоординатный подъём": w_cd,
        "логит на A: отзывы+описание": w_lA,
        "логит на A без истории: отзывы+описание": w_lAn,
        "логит на A без истории: + рейтинг": w_lAn3,
        "логит на A без истории: все признаки": w_lAall,
    }
    vrows = []
    for name, ww in variants.items():
        r = ev.recall(ww)
        dB, dBn, dN = (boot((r - base)[m].to_numpy()) for m in (ib, ib & nh, nh))
        vrows.append((name, str({k: round(v, 4) for k, v in ww.items()}), r[ia].mean(), r[ib].mean(), *dB,
                      r[ia & nh].mean(), r[ib & nh].mean(), *dBn, *dN, r.mean()))
    vt = pd.DataFrame(vrows, columns=["вариант", "веса", "A", "B", "dB", "loB", "hiB", "A_nh", "B_nh", "dB_nh", "loB_nh",
                                     "hiB_nh", "d_nh3000", "lo_nh3000", "hi_nh3000", "все"])
    pd.set_option("display.width", 250)
    print(vt.round(4).to_string(index=False))
    vt.to_csv(WORK / "quality_variants.csv", index=False)
    # Выбор: на бенчмарке почти все ответы «без истории», а на срезе с историей эффект раздут
    # (отзывы и длина описания — косвенный признак популярности). Поэтому берём веса, подобранные
    # на A без истории гладкой целью (условный логит): они вдвое мягче, меньше рискуют и не хуже на B без истории.
    best_name = "логит на A без истории: отзывы+описание"
    wbest = variants[best_name]
    print(f"выбрано: {best_name} {wbest}")
    for s in (0.5, 0.75, 1.0, 1.5, 2.0):
        r = ev.recall({k: v * s for k, v in wbest.items()})
        print(f"  масштаб {s}: A {r[ia].mean():.4f}  B {r[ib].mean():.4f}  B без истории {r[ib & nh].mean():.4f}  все {r.mean():.4f}")
    # ожидание на бенчмарке: p·Δ(история) + (1 − p)·Δ(без истории), p ≈ 0.04 (eda/05)
    r = ev.recall(wbest)
    dh, dn = (r - base)[~nh].mean(), (r - base)[nh].mean()
    print(f"  Δ история {dh:+.4f}, без истории {dn:+.4f}; ожидание на бенчмарке при p=0.04: {0.04 * dh + 0.96 * dn:+.4f}")

    # === 3. Срезы и перемещения ===
    rb = ev.recall(wbest)
    print("\n=== 3. Срезы (все 3000) ===")
    sl = pd.concat([slices(base, vq)["recall"].rename("база"), slices(rb, vq)["recall"].rename("качество"),
                    slices(base, vq)["запросов"].rename("n")], axis=1)
    sl["Δ"] = sl["качество"] - sl["база"]
    # бутстреп Δ на срезах
    d = (rb - base)
    dq = vq.set_index("query_id").loc[d.index]
    ci = {}
    for name, m in {"все": np.ones(len(d), bool), "текст есть в train": dq.text_in_train.to_numpy(),
                    "текста нет в train": ~dq.text_in_train.to_numpy(),
                    "ответ без истории": ~dq.target_has_history.to_numpy(),
                    "у ответа есть история": dq.target_has_history.to_numpy(),
                    "без фильтра": ~dq.has_filter.to_numpy(), "с фильтром": dq.has_filter.to_numpy(),
                    "регион или вся Россия": dq.is_region.to_numpy(), "поиск по городу": ~dq.is_region.to_numpy()}.items():
        _, lo, hi = boot(d[m].to_numpy())
        ci[name] = f"[{lo:+.4f}; {hi:+.4f}]"
    sl["95% ДИ"] = pd.Series(ci)
    print(sl.round(4).to_string())
    up = ((rb > base)).sum(), ((rb < base)).sum()
    print(f"запросов, где правильный ответ поднят в топ-50: {up[0]}, выбит из топ-50: {up[1]}")
    for k in (10, 20, 100):
        print(f"  Recall@{k}: {ev.recall({}, k).mean():.4f} -> {ev.recall(wbest, k).mean():.4f}")

    # === 4. Сдвиг признаков: топ-50 валидации vs бенчмарка ===
    cb = pd.read_parquet(CANDS / "hybrid_geo2_meta_bench.parquet")
    fb = rq.candidate_features(cb, qf)
    cols = ["rev_log", "has_rev", "has_rating", "rating_c", "price_stub", "phone_hidden", "msg_forbidden",
            "desc_log", "title_len"]
    tv, tb = fv[fv["rank"] <= K], fb[fb["rank"] <= K]
    shift = pd.DataFrame({"вал топ-50": tv[cols].mean(), "бенч топ-50": tb[cols].mean(),
                          "вал 51-300": fv.loc[fv["rank"] > K, cols].mean(), "бенч 51-300": fb.loc[fb["rank"] > K, cols].mean()})
    # внутризапросный разброс отзывов: std pct не меняется, смотрим std сырого признака в запросе
    shift.loc["std rev_log в запросе"] = [tv.groupby("query_id").rev_log.std().mean(), tb.groupby("query_id").rev_log.std().mean(),
                                          np.nan, np.nan]
    print("\n=== 4. Сдвиг признаков ===")
    print(shift.round(3).to_string())

    # === 5. Сохранение ===
    nv = rq.rerank_quality(fv, None, wbest)
    nb = rq.rerank_quality(fb, None, wbest)
    assert nb.groupby("query_id").size().eq(cb.groupby("query_id").size()).all()
    chk = ev.recall(wbest).mean()
    from validation import recall_at_k
    r_saved = recall_at_k(nv[nv["rank"] <= K].groupby("query_id")["item_id"].agg(list).to_dict(), labels).mean()
    print(f"\nпроверка: быстрый {chk:.4f}, по сохранённому файлу {r_saved:.4f}")
    nv.to_parquet(CANDS / "hybrid_geo2_meta_q_val.parquet", index=False)
    nb.to_parquet(CANDS / "hybrid_geo2_meta_q_bench.parquet", index=False)
    for name, old, new in (("вал", cv, nv), ("бенч", cb, nb)):
        same = new[new["rank"] <= K].merge(old[old["rank"] <= K], on=["query_id", "item_id"]).shape[0] / (old["rank"] <= K).sum()
        print(f"{name}: топ-50 совпадает с исходным на {same:.3f}")
    print(f"{time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
