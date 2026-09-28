"""Разбор промахов лучшего решения (hybrid_geo2_meta_val, Recall@50 = 0.869). Только анализ, без моделей.

Запуск из корня репозитория: python experiments/error_analysis.py  (~1 мин, CPU, пик ~2.5 ГБ)
Пишет work/errors/*.parquet|csv; выводы — eda/14_errors.md.

Что считаем:
1. Промахи (ни один правильный ответ не в топ-50) по срезам и где лежит лучший правильный ответ: 51–100, 101–300, >300.
2. Для промахов с ответом в 51–300: правильный ответ против топ-50 того же запроса (попарная доля побед
   P(ответ > сосед) + 0.5·P(равно), усреднённая по запросам) по мета-признакам. Контраст — попадания против соседей.
3. Промахи вне топ-300: грубая автоклассификация причин + примеры.
4. «Близнецы»: в топ-50 есть другой item с тем же нормализованным заголовком и той же локацией.
5. Чувствительность: +w·признак к score, w выбирается на половине A, проверяется на B (не модель — оценка потолка сигнала).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from validation import DATA, WORK, load_validation  # noqa: E402
from rerank_meta import core_filters  # noqa: E402

OUT = WORK / "errors"
OUT.mkdir(exist_ok=True)
CANDS = WORK / "cands" / "hybrid_geo2_meta_val.parquet"
RUSSIA = 621540
_NONW = re.compile(r"[^0-9a-zа-я]+")


def norm_title(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower().str.replace("ё", "е", regex=False)
    return s.str.replace(_NONW, " ", regex=True).str.strip()


def tokens5(s: str) -> set:
    return {w[:5] for w in _NONW.sub(" ", s.lower().replace("ё", "е")).split() if len(w) >= 2}


def load_items() -> pd.DataFrame:
    cols = ["item_id", "item_title_raw", "item_infm_params_text", "item_microcat_id", "item_location_id",
            "item_rating", "item_rating_reviews_count", "item_price", "item_is_phone_hidden",
            "item_is_message_forbidden", "item_latitude", "item_longitude"]
    it = pd.read_parquet(DATA / "benchmark_items.parquet", columns=cols)
    d = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_description_raw"])["item_description_raw"]
    it["desc_len"] = d.fillna("").str.len().to_numpy()
    it["desc_head"] = d.fillna("").str.slice(0, 300).to_numpy()  # для примеров и дублей
    del d
    it["price"] = pd.to_numeric(it["item_price"].astype(str), errors="coerce")
    it["lat"] = it["item_latitude"].astype(float)
    it["lon"] = it["item_longitude"].astype(float)
    it = it.drop(columns=["item_price", "item_latitude", "item_longitude"])
    it["title_norm"] = norm_title(it["item_title_raw"])
    it["reviews"] = it["item_rating_reviews_count"].fillna(0)
    it["has_rating"] = it["item_rating"].notna() & (it["item_rating"] > 0)
    it["rating"] = it["item_rating"].fillna(0)
    it["price_stub"] = it["price"].isna() | it["price"].isin([-1, 0, 1])
    it["log_price"] = np.log1p(it["price"].where(~it["price_stub"]).fillna(0))
    it["phone_hidden"] = it["item_is_phone_hidden"].astype(bool)
    it["msg_forbidden"] = it["item_is_message_forbidden"].astype(bool)
    return it


def main():
    vq, labels, mask = load_validation()
    vq = vq.set_index("query_id")
    vq["slice_geo"] = np.where(~vq["is_region"], "город",
                               np.where(vq["search_location_id"] == RUSSIA, "621540", "регион"))
    c = pd.read_parquet(CANDS)
    it = load_items()
    idx = pd.Series(np.arange(len(it)), index=it["item_id"])

    # история (клики в train[mask]) — завышена на валидации, смотрим отдельно
    tr = pd.read_parquet(DATA / "train.parquet", columns=["item_id"])["item_id"][mask]
    hist = tr.value_counts()
    it["hist"] = it["item_id"].map(hist).fillna(0).to_numpy()
    del tr

    # центроиды локаций
    cent = it.groupby("item_location_id")[["lat", "lon"]].median()

    c["pos"] = idx.reindex(c["item_id"]).to_numpy()
    gold = [(q, g) for q, s in labels.items() for g in s]
    gdf = pd.DataFrame(gold, columns=["query_id", "item_id"])
    gdf = gdf.merge(c[["query_id", "item_id", "rank"]], how="left")
    gdf["rank"] = gdf["rank"].fillna(999).astype(int)
    c["is_gold"] = pd.MultiIndex.from_frame(c[["query_id", "item_id"]]).isin(
        pd.MultiIndex.from_frame(gdf[["query_id", "item_id"]]))

    # --- 1. промахи по срезам
    rec = gdf.assign(hit=gdf["rank"] <= 50).groupby("query_id")["hit"].mean()
    best = gdf.groupby("query_id")["rank"].min()
    q = vq.loc[rec.index].copy()
    q["recall"], q["best_rank"] = rec, best
    q["miss"] = q["best_rank"] > 50
    q["bucket"] = pd.cut(q["best_rank"], [0, 50, 100, 300, 10_000], labels=["1–50", "51–100", "101–300", ">300"])
    print("Recall@50", q["recall"].mean(), "промахов", q["miss"].sum(), "потеря", (1 - q["recall"]).mean())

    def slice_tab(q):
        rows = {}
        groups = {"все": q.index == q.index, "город": q["slice_geo"] == "город", "регион": q["slice_geo"] == "регион",
                  "621540": q["slice_geo"] == "621540", "текст знакомый": q["text_in_train"],
                  "текст новый": ~q["text_in_train"], "с фильтром": q["has_filter"], "без фильтра": ~q["has_filter"],
                  "ответ с историей": q["target_has_history"], "ответ без истории": ~q["target_has_history"]}
        for n, m in groups.items():
            s = q[m]
            b = s["bucket"].value_counts()
            rows[n] = {"запросов": len(s), "recall@50": s["recall"].mean(), "промахов": int(s["miss"].sum()),
                       "доля промахов": s["miss"].mean(),
                       "51–100": int(b.get("51–100", 0)), "101–300": int(b.get("101–300", 0)), ">300": int(b.get(">300", 0)),
                       "потеря п.п. (от всех 3000)": 100 * (1 - s["recall"]).sum() / len(q)}
        return pd.DataFrame(rows).T
    t1 = slice_tab(q)
    print(t1.to_string())
    t1.to_csv(OUT / "t1_slices.csv")

    # --- признаки кандидатов
    feats = ["reviews", "rating", "has_rating", "price_stub", "log_price", "phone_hidden", "msg_forbidden",
             "desc_len", "hist"]
    F = it[feats].to_numpy(np.float64)
    loc = it["item_location_id"].to_numpy()
    mc = it["item_microcat_id"].to_numpy()
    params = it["item_infm_params_text"].fillna("").to_numpy()
    titles = it["title_norm"].to_numpy()

    top = c[c["rank"] <= 50]
    top10 = c[c["rank"] <= 10]
    mode_mc = top10.assign(mc=mc[top10["pos"]]).groupby("query_id")["mc"].agg(lambda s: s.mode().iloc[0])
    mode_loc = top10.assign(l=loc[top10["pos"]]).groupby("query_id")["l"].agg(lambda s: s.mode().iloc[0])

    def extra(qid, pos_arr):
        """Признаки, зависящие от запроса: тот же город, мк как у топ-10, фильтр, слова запроса в заголовке."""
        row = vq.loc[qid]
        l = loc[pos_arr]
        same_city = (l == row["search_location_id"]) if not row["is_region"] else (l == mode_loc[qid])
        same_mc = mc[pos_arr] == mode_mc[qid]
        f = core_filters(row["search_infm_params_text"] or "")
        subs = f["vid"] + f["tip"]
        fm = np.array([all(any(v in params[p] for v in f[k]) for k in ("vid", "tip") if f[k]) for p in pos_arr]) \
            if subs else np.full(len(pos_arr), np.nan)
        qt = tokens5(row["search_query"])
        tw = np.array([len(qt & tokens5(titles[p])) / max(len(qt), 1) for p in pos_arr])
        return np.column_stack([same_city, same_mc, fm, tw])

    names = feats + ["same_city", "same_mc_top10", "filter_match", "q_words_in_title"]

    def compare(qids, gold_of):
        """Для каждого запроса: признаки ответа, средние соседей из топ-50 и попарная доля побед."""
        out = []
        tg = top.groupby("query_id")
        for qid in qids:
            gpos = gold_of[qid]
            nb = tg.get_group(qid)
            nb = nb[~nb["is_gold"]]["pos"].to_numpy()
            if len(nb) == 0:
                continue
            fg = np.concatenate([F[[gpos]], extra(qid, np.array([gpos]))], axis=1)[0]
            fn = np.concatenate([F[nb], extra(qid, nb)], axis=1)
            win = []
            for j in range(len(names)):
                a, b = fg[j], fn[:, j]
                if np.isnan(a):
                    win.append(np.nan)
                    continue
                win.append(((a > b).sum() + 0.5 * (a == b).sum()) / len(b))
            out.append(np.concatenate([fg, np.nanmean(fn, axis=0), win]))
        cols = [f"g_{n}" for n in names] + [f"n_{n}" for n in names] + [f"w_{n}" for n in names]
        return pd.DataFrame(out, columns=cols)

    # лучший правильный ответ запроса (минимальный ранг)
    gbest = gdf.sort_values("rank").drop_duplicates("query_id").set_index("query_id")
    gbest["pos"] = idx.reindex(gbest["item_id"]).to_numpy()
    mid = q.index[(q["best_rank"] > 50) & (q["best_rank"] <= 300)]
    hit = q.index[q["best_rank"] <= 50]
    cm = compare(mid, gbest["pos"])
    ch = compare(hit, gbest["pos"])

    def summ(df):
        r = {}
        for n in names:
            r[n] = {"ответ": np.nanmean(df[f"g_{n}"]), "соседи": np.nanmean(df[f"n_{n}"]),
                    "доля побед": np.nanmean(df[f"w_{n}"]), "n": int(df[f"w_{n}"].notna().sum())}
        return pd.DataFrame(r).T
    s_mid, s_hit = summ(cm), summ(ch)
    t2 = pd.concat({"промахи 51–300": s_mid, "попадания": s_hit}, axis=1)
    # медианы для отзывов и длины описания
    for n in ["reviews", "desc_len", "hist"]:
        t2.loc[n + " (медиана)"] = [np.nanmedian(cm[f"g_{n}"]), np.nanmedian(cm[f"n_{n}"]), np.nan, len(cm),
                                    np.nanmedian(ch[f"g_{n}"]), np.nanmedian(ch[f"n_{n}"]), np.nan, len(ch)]
    print(t2.round(3).to_string())
    t2.to_csv(OUT / "t2_signals.csv")

    # --- 4. близнецы: в топ-50 другой item с тем же заголовком и той же локацией
    ttl = it["title_norm"].to_numpy()
    dh = it["desc_head"].to_numpy()
    miss = q.index[q["miss"]]
    tw_rows = []
    tg = top.groupby("query_id")
    for qid in miss:
        gps = [idx[g] for g in labels[qid]]
        nb = tg.get_group(qid)["pos"].to_numpy()
        t_c = any(((ttl[nb] == ttl[g]) & (loc[nb] == loc[g])).any() for g in gps)
        t_any = any((ttl[nb] == ttl[g]).any() for g in gps)
        t_desc = any(((ttl[nb] == ttl[g]) & (loc[nb] == loc[g]) & (dh[nb] == dh[g])).any() for g in gps)
        d_c = any(((dh[nb] == dh[g]) & (loc[nb] == loc[g]) & (len(dh[g]) >= 50)).any() for g in gps)
        tw_rows.append((qid, t_any, t_c, t_desc, d_c))
    tw = pd.DataFrame(tw_rows, columns=["query_id", "twin_title", "twin_title_city", "twin_title_city_desc",
                                        "twin_desc_city"]).set_index("query_id")
    q = q.join(tw)
    print("близнецы среди промахов:", tw.sum().to_dict(), "из", len(tw))
    print(q[q["miss"]].groupby("bucket", observed=True)[["twin_title", "twin_title_city", "twin_title_city_desc", "twin_desc_city"]].sum())
    # для контраста: сколько «близнецов» у попаданий (ответ в топ-50 и его двойник тоже в топ-50)
    hh = []
    for qid in hit[:3000]:
        g = gbest.loc[qid, "pos"]
        nb = tg.get_group(qid)["pos"].to_numpy()
        nb = nb[nb != g]
        hh.append(((ttl[nb] == ttl[g]) & (loc[nb] == loc[g])).any())
    print("у попаданий есть близнец в топ-50:", np.mean(hh))

    # --- 3. промахи вне 300: автоклассификация
    far = q.index[q["best_rank"] > 300]
    rows = []
    for qid in far:
        row = vq.loc[qid]
        g = gbest.loc[qid, "pos"]
        qt = tokens5(row["search_query"])
        in_title = len(qt & tokens5(ttl[g])) > 0
        in_any = len(qt & tokens5(ttl[g] + " " + params[g] + " " + dh[g])) > 0
        ml = mode_loc[qid]
        if row["is_region"]:
            geo_bad = loc[g] != ml
        else:
            geo_bad = loc[g] != row["search_location_id"]
        ref = row["search_location_id"] if not row["is_region"] else ml
        try:
            a, b = cent.loc[loc[g]], cent.loc[ref]
            dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin(np.radians(b.lat - a.lat) / 2) ** 2 + np.cos(np.radians(a.lat))
                                                * np.cos(np.radians(b.lat)) * np.sin(np.radians(b.lon - a.lon) / 2) ** 2))
        except KeyError:
            dist = np.nan
        f = core_filters(row["search_infm_params_text"] or "")
        fok = all(any(v in params[g] for v in f[k]) for k in ("vid", "tip") if f[k]) if (f["vid"] or f["tip"]) else None
        top3 = c[(c["query_id"] == qid) & (c["rank"] <= 3)].sort_values("rank")["pos"].to_numpy()
        rows.append({"query_id": qid, "query": row["search_query"], "filter": row["search_infm_params_text"],
                     "loc": row["search_location_id"], "geo": row["slice_geo"], "gold_title": it["item_title_raw"].iat[g],
                     "gold_loc": loc[g], "dist_km": dist, "geo_bad": geo_bad, "q_in_title": in_title, "q_in_any": in_any,
                     "same_mc_top10": mc[g] == mode_mc[qid], "filter_ok": fok, "twin": q.loc[qid, "twin_title_city"],
                     "text_in_train": row["text_in_train"],
                     "top3": " | ".join(it["item_title_raw"].iat[p] for p in top3)})
    fr = pd.DataFrame(rows)

    def cause(r):
        if r["twin"]:
            return "близнец в топ-50"
        if r["geo_bad"] and (np.isnan(r["dist_km"]) or r["dist_km"] > 30):
            return "другой город (>30 км)"
        if r["filter_ok"] is False:
            return "ответ не проходит фильтр"
        if not r["q_in_any"]:
            return "нет общих слов (лексика/синоним)"
        if not r["same_mc_top10"]:
            return "другая микрокатегория (другая услуга по смыслу)"
        if not r["q_in_title"]:
            return "слова только в описании/параметрах"
        return "та же тема, текст совпадает (равнозначные)"
    fr["cause"] = fr.apply(cause, axis=1)
    print(fr["cause"].value_counts().to_string())
    print(fr.groupby("geo")["cause"].value_counts().unstack(fill_value=0).to_string())
    fr.to_csv(OUT / "far_misses.csv", index=False)

    # --- 5. чувствительность: score + w·признак на промахах/попаданиях, выбор на A, проверка на B
    rng = np.random.default_rng(0)
    qids = np.array(sorted(labels))
    half_a = set(rng.choice(qids, len(qids) // 2, replace=False))
    cc = c.copy()
    P = cc["pos"].to_numpy()
    cand_feats = {
        "log1p(reviews)": np.log1p(F[P, 0]),
        "has_rating": F[P, 2],
        "price_stub": F[P, 3],
        "phone_hidden": F[P, 5],
        "msg_forbidden": F[P, 6],
        "log1p(desc_len)": np.log1p(F[P, 7]),
        "log1p(hist) [завышено]": np.log1p(F[P, 8]),
    }
    base = cc["score"].to_numpy(np.float64)
    qa = cc["query_id"].isin(half_a).to_numpy()

    def recall_of(s):
        tmp = pd.DataFrame({"q": cc["query_id"].to_numpy(), "s": s, "g": cc["is_gold"].to_numpy()})
        tmp["r"] = tmp.groupby("q")["s"].rank(ascending=False, method="first")
        hits = tmp[(tmp["r"] <= 50) & tmp["g"]].groupby("q").size()
        n_g = {k: len(v) for k, v in labels.items()}
        r = pd.Series({k: hits.get(k, 0) / n_g[k] for k in labels})
        return r
    r0 = recall_of(base)
    ina = r0.index.isin(list(half_a))
    sens = []
    for name, x in cand_feats.items():
        best = (0.0, r0[ina].mean(), r0[~ina].mean(), 0.0)
        for w in [-0.004, -0.002, -0.001, -0.0005, 0.0005, 0.001, 0.002, 0.004]:
            r = recall_of(base + w * x)
            a = r[ina].mean()
            if a > best[1]:
                best = (w, a, r[~ina].mean(), r.mean())
        w, a, b, al = best
        sens.append({"признак": name, "w (по A)": w, "A": a, "B": b, "Δ B": b - r0[~ina].mean(),
                     "все": al if w else r0.mean()})
    sens = pd.DataFrame(sens)
    print("база A/B", r0[ina].mean(), r0[~ina].mean())
    print(sens.round(4).to_string())
    sens.to_csv(OUT / "t5_sensitivity.csv", index=False)

    # 5b. отзывы + длина описания вместе (сетка по A), срезы для выбранных весов
    xr, xd = cand_feats["log1p(reviews)"], cand_feats["log1p(desc_len)"]
    grid = []
    for wr in [0, 0.0005, 0.001, 0.0015, 0.002]:
        for wd in [0, 0.001, 0.002, 0.003]:
            r = recall_of(base + wr * xr + wd * xd)
            grid.append((wr, wd, r[ina].mean(), r[~ina].mean(), r.mean()))
    grid = pd.DataFrame(grid, columns=["w_rev", "w_desc", "A", "B", "все"])
    print(grid.round(4).to_string())
    grid.to_csv(OUT / "t5_grid.csv", index=False)
    wr, wd = grid.sort_values("A", ascending=False).iloc[0][["w_rev", "w_desc"]]
    variants = {"база": r0, f"отзывы {wr}+описание {wd}": recall_of(base + wr * xr + wd * xd),
                "только отзывы 0.001": recall_of(base + 0.001 * xr),
                "история 0.004 [завышено]": recall_of(base + 0.004 * cand_feats["log1p(hist) [завышено]"])}
    qq = q.loc[r0.index]
    groups = {"все": qq.index == qq.index, "половина B": ~ina, "город": qq["slice_geo"] == "город",
              "регион": qq["slice_geo"] == "регион", "621540": qq["slice_geo"] == "621540",
              "текст новый": ~qq["text_in_train"], "ответ с историей": qq["target_has_history"],
              "ответ без истории": ~qq["target_has_history"]}
    sl = pd.DataFrame({v: {g: rr[np.asarray(m)].mean() for g, m in groups.items()} for v, rr in variants.items()})
    print(sl.round(4).to_string())
    sl.to_csv(OUT / "t5_slices.csv")
    # сколько промахов 51–300 спасает выбранный вариант и сколько попаданий теряет
    rb = variants[f"отзывы {wr}+описание {wd}"]
    print("спасено", int(((r0 == 0) & (rb > 0)).sum()), "потеряно", int(((r0 > 0) & (rb == 0)).sum()))

    q.reset_index().to_parquet(OUT / "per_query.parquet", index=False)


if __name__ == "__main__":
    main()
