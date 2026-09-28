"""Обучаемый реранкер (learning-to-rank) поверх кандидатов geo2 (см. eda/16_ltr.md).

Идея: вместо четырёх ручных бонусов (eda/11_meta_boost.md) модель по признакам пары «запрос–кандидат»
сама решает, кого из топ-300 geo2 поднять в топ-50. Порядок внутри 50 для Recall@50 не важен,
поэтому итог — сортировка кандидатов по предсказанию модели.

Шаги (запуск из корня репозитория, только CPU):
    python src/ltr.py sources   # скоры источников для кандидатов: BM25 (lemma+stem) и косинус bge-m3,
                                # глобальные ранги в корпусе -> work/ltr/src_{val,bench}.parquet (~6–8 мин)
    python src/ltr.py run       # признаки, кросс-валидация по запросам (2 и 5 фолдов), модели,
                                # важности, бутстреп, запись work/cands/hybrid_geo2_ltr_{val,bench}.parquet

Против утечки:
    - классификатор микрокатегорий и гео-доли «локация поиска → локация объявления» для валидации строятся
      по train[train_mask], для бенчмарка — по полному train;
    - модель для валидации учится на одной части запросов и предсказывает другую (вне-фолдовые скоры),
      для бенчмарка — на всех 3000 запросах валидации;
    - признаков популярности / истории объявления нет намеренно: на валидации у 53% ответов есть история,
      на бенчмарке заметно меньше (eda/05_validation.md), модель бы переучилась на неё.
"""
from __future__ import annotations

import argparse
import re
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
CANDS = WORK / "cands"
LTR = WORK / "ltr"
PY_SEED = 0


# ====================================================================== скоры источников

def _sources_for(cands: pd.DataFrame, queries: pd.DataFrame, pos_of: dict, sp, D: np.ndarray,
                 q_emb: np.ndarray) -> pd.DataFrame:
    """Для каждой пары (запрос, кандидат): BM25-скор и глобальный ранг BM25 в корпусе,
    косинус bge-m3 и глобальный ранг по косинусу. cands — в порядке queries, по 300 на запрос."""
    qids = queries["query_id"].to_numpy()
    texts = queries["search_query"].fillna("").astype(str).to_numpy()
    g = cands.groupby("query_id", sort=False).indices
    n = len(cands)
    bm, bm_r, cs, cs_r = (np.zeros(n, np.float32), np.zeros(n, np.int32),
                          np.zeros(n, np.float32), np.zeros(n, np.int32))
    cpos = cands["item_id"].map(pos_of).to_numpy()
    assert not np.isnan(cpos.astype(float)).any()
    cpos = cpos.astype(np.int64)
    bs = 64
    t0 = time.time()
    for b0 in range(0, len(qids), bs):
        dsc = q_emb[b0:b0 + bs] @ D.T                       # [bs, 189k] косинусы
        dsort = np.sort(dsc, axis=1)
        for j in range(dsc.shape[0]):
            i = b0 + j
            rows = g[qids[i]]
            p = cpos[rows]
            # dense: косинус и глобальный ранг (1 + число объявлений с большим косинусом)
            v = dsc[j, p]
            cs[rows] = v
            cs_r[rows] = D.shape[0] - np.searchsorted(dsort[j], v, side="right") + 1
            # BM25: скор и глобальный ранг среди всего корпуса
            s = sp.scores(texts[i])
            ss = np.sort(s)
            sv = s[p]
            bm[rows] = sv
            bm_r[rows] = len(s) - np.searchsorted(ss, sv, side="right") + 1
        if (b0 // bs) % 10 == 0:
            print(f"  {min(b0 + bs, len(qids))}/{len(qids)} за {time.time() - t0:.0f} с", flush=True)
    return pd.DataFrame({"query_id": cands["query_id"].to_numpy(), "item_id": cands["item_id"].to_numpy(),
                         "bm25": bm, "bm25_grank": bm_r, "cos": cs, "cos_grank": cs_r})


def cmd_sources():
    """Считает скоры источников для валидации и бенчмарка (CPU) и кеширует их в work/ltr/."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # только CPU: GPU занят
    import torch
    torch.set_num_threads(8)
    from dense import encode_queries, load_corpus_emb, load_model
    from sparse import load_items
    from sparse_lemma import LemmaIndex
    from validation import load_validation

    LTR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vq, _, _ = load_validation()
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    for q in (vq, bq):
        q["query_id"] = q["query_id"].astype(str)
    # 1) запросы -> bge-m3 на CPU в fp32 (без int8: он портит качество, а запросы короткие)
    model = load_model("bge-m3", 256, quantize=False)
    qv = encode_queries(model, "bge-m3", vq, with_filters=True, batch_size=64)
    print(f"bge-m3 валидация: {qv.shape} за {time.time() - t0:.0f} с", flush=True)
    qb = encode_queries(model, "bge-m3", bq, with_filters=True, batch_size=64)
    print(f"bge-m3 бенчмарк: {qb.shape} за {time.time() - t0:.0f} с", flush=True)
    del model
    np.save(LTR / "q_bge_val.npy", qv)
    np.save(LTR / "q_bge_bench.npy", qb)
    # 2) корпус
    emb, ids = load_corpus_emb("bge-m3_s256")
    D = emb.astype(np.float32)
    del emb
    pos_of = {x: i for i, x in enumerate(ids)}
    items = load_items()
    assert (items["item_id"].astype(str).to_numpy() == ids).all()
    t = time.time()
    sp = LemmaIndex.build(items, scheme="lemma+stem", variant="b")
    print(f"BM25 lemma+stem за {time.time() - t:.0f} с", flush=True)
    del items
    for split, q, qe in (("val", vq, qv), ("bench", bq, qb)):
        c = pd.read_parquet(CANDS / f"hybrid_geo2_{split}.parquet")
        c = c.set_index("query_id").loc[q["query_id"]].reset_index()   # порядок как в queries
        out = _sources_for(c, q, pos_of, sp, D, qe)
        out.to_parquet(LTR / f"src_{split}.parquet", index=False)
        print(f"{split}: {len(out)} строк за {time.time() - t0:.0f} с", flush=True)




# ====================================================================== признаки

_TOK = re.compile(r"[а-яёa-z0-9]+")


def _tokens(s: str) -> list[str]:
    """Грубые «стемы» для перекрытия с заголовком: нижний регистр, ё→е, первые 5 символов слова."""
    return [t[:5] for t in _TOK.findall(str(s).lower().replace("ё", "е")) if len(t) > 1]


def _in_query(df: pd.DataFrame, col: str, how: str) -> np.ndarray:
    """Внутри-запросная нормировка: "pct" — перцентиль (1 = лучший), "max" — отношение к максимуму,
    "gap" — разница с максимумом, "z" — z-оценка."""
    g = df.groupby("query_id", sort=False)[col]
    if how == "pct":
        return g.rank(pct=True, method="average").to_numpy(np.float32)
    if how == "max":
        return (df[col] / g.transform("max").replace(0, np.nan)).fillna(0).to_numpy(np.float32)
    if how == "gap":
        return (df[col] - g.transform("max")).to_numpy(np.float32)
    if how == "z":
        return ((df[col] - g.transform("mean")) / (g.transform("std") + 1e-6)).to_numpy(np.float32)
    raise ValueError(how)


class GeoCtx:
    """Гео-контекст для признаков: доли «локация поиска → локация объявления» по train_fit,
    центроиды локаций (по координатам объявлений, это не ответы) и «главный город» региона."""

    def __init__(self, train_fit: pd.DataFrame, cent: pd.DataFrame, regions: set):
        t = train_fit[["search_location_id", "item_location_id"]]
        c = t.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        c["share"] = c["n"] / c.groupby("search_location_id")["n"].transform("sum")
        self.share = c.set_index(["search_location_id", "item_location_id"])["share"]
        # главный город региона = локация с наибольшей долей выборов (для расстояния у региональных запросов)
        top = c.sort_values("share", ascending=False).drop_duplicates("search_location_id")
        self.main_city = dict(zip(top["search_location_id"], top["item_location_id"]))
        self.cent = cent
        self.regions = regions


def add_sources(f: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """Признаки источников: BM25 и косинус bge-m3, их глобальные ранги в корпусе и внутри-запросные нормировки."""
    f = f.merge(src, on=["query_id", "item_id"], how="left", validate="1:1")
    assert f["bm25"].notna().all(), "нет скоров источников для части кандидатов"
    f["bm25_grank"] = np.log1p(f["bm25_grank"]).astype(np.float32)
    f["cos_grank"] = np.log1p(f["cos_grank"]).astype(np.float32)
    f["bm25_pct"] = _in_query(f, "bm25", "pct")
    f["bm25_max"] = _in_query(f, "bm25", "max")
    f["cos_pct"] = _in_query(f, "cos", "pct")
    f["cos_gap"] = _in_query(f, "cos", "gap")
    f["cos_z"] = _in_query(f, "cos", "z")
    f["bm25_zero"] = (f["bm25"] <= 0).astype(np.int8)
    return f


def build_features(cands: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame, src: pd.DataFrame | None,
                   mc_model, geo: GeoCtx) -> pd.DataFrame:
    """Признаки пар «запрос–кандидат» (топ-300 geo2).

    cands   — query_id, item_id, score, rank (кандидаты geo2);
    queries — query_id, search_query, search_location_id, search_infm_params_text, search_is_delivery_search,
              text_in_train (для бенчмарка флаг по полному train);
    items   — объявления корпуса (item_id, заголовок, описание, параметры, микрокатегория, локация, координаты,
              рейтинг, отзывы, цена, флаги);
    src     — скоры источников из cmd_sources (bm25, bm25_grank, cos, cos_grank);
    mc_model — rerank_meta.fit_microcat(train_fit); geo — GeoCtx(train_fit, ...).
    Возвращает кандидатов (query_id, item_id, rank) + колонки признаков; порядок строк — как в cands."""
    import rerank_meta as rm
    t0 = time.time()
    f = cands[["query_id", "item_id", "score", "rank"]].reset_index(drop=True).copy()

    # --- geo2: исходный ранг и скор, нормировки
    f["g_rank"] = f["rank"].astype(np.float32)
    f["g_score"] = f["score"].astype(np.float32)
    f["g_score_max"] = _in_query(f, "score", "max")
    s50 = f[f["rank"] == 50].set_index("query_id")["score"]
    f["g_gap50"] = (f["score"] - f["query_id"].map(s50).fillna(0)).astype(np.float32)

    # --- источники (если скоры уже посчитаны; иначе их добавляет add_sources отдельно)
    if src is not None:
        f = add_sources(f, src)

    # --- объявление: качество и тексты
    it = items.set_index("item_id")
    ii = it.reindex(f["item_id"])
    rev = ii["item_rating_reviews_count"].to_numpy(float)
    f["log_reviews"] = np.log1p(np.nan_to_num(rev, nan=0)).astype(np.float32)
    f["rating"] = ii["item_rating"].to_numpy(np.float32)                 # NaN оставляем: деревья умеют
    f["has_rating"] = ii["item_rating"].notna().to_numpy().astype(np.int8)
    price = ii["item_price"].astype(float).to_numpy()
    f["price_stub"] = (np.nan_to_num(price, nan=-1) <= 1).astype(np.int8)   # -1 / 0 / 1 = «цена не указана»
    f["log_price"] = np.where(price > 1, np.log1p(np.clip(price, 0, 1e9)), np.nan).astype(np.float32)
    f["phone_hidden"] = ii["item_is_phone_hidden"].fillna(False).to_numpy().astype(np.int8)
    f["msg_forbidden"] = ii["item_is_message_forbidden"].fillna(False).to_numpy().astype(np.int8)
    f["desc_len"] = np.log1p(ii["item_description_raw"].fillna("").str.len().to_numpy()).astype(np.float32)
    f["title_len"] = ii["item_title_raw"].fillna("").str.len().to_numpy().astype(np.float32)

    # перекрытие слов запроса с заголовком (грубые стемы)
    qtok = {q: set(_tokens(s)) for q, s in zip(queries["query_id"], queries["search_query"])}
    ttok = {i: set(_tokens(s)) for i, s in zip(it.index, it["item_title_raw"].fillna(""))}
    ov = np.zeros(len(f), np.float32)
    for k, (q, i) in enumerate(zip(f["query_id"].to_numpy(), f["item_id"].to_numpy())):
        a = qtok.get(q)
        if a:
            ov[k] = len(a & ttok.get(i, set())) / len(a)
    f["title_overlap"] = ov

    # --- запрос
    qq = queries.set_index("query_id").reindex(f["query_id"])
    f["q_words"] = qq["search_query"].fillna("").str.split().str.len().to_numpy().astype(np.float32)
    f["q_has_filter"] = qq["search_infm_params_text"].fillna("").str.strip().ne("").to_numpy().astype(np.int8)
    f["q_text_in_train"] = qq["text_in_train"].astype(bool).to_numpy().astype(np.int8)
    f["q_delivery"] = qq["search_is_delivery_search"].fillna(0).to_numpy().astype(np.int8)
    sloc = qq["search_location_id"].to_numpy()
    f["q_region"] = np.isin(sloc, list(geo.regions)).astype(np.int8)
    f["q_russia"] = (sloc == 621540).astype(np.int8)

    # --- гео
    iloc = ii["item_location_id"].to_numpy()
    f["same_city"] = (iloc == sloc).astype(np.int8)
    key = pd.MultiIndex.from_arrays([sloc, iloc])
    f["loc_share"] = geo.share.reindex(key).fillna(0).to_numpy(np.float32)
    f["loc_share_log"] = np.log10(f["loc_share"] + 1e-4).astype(np.float32)
    # расстояние от объявления до центроида локации поиска (для региона — до его главного города)
    anchor = np.array([l if l not in geo.regions else geo.main_city.get(l, -1) for l in sloc])
    c = geo.cent.reindex(anchor)
    lat = pd.to_numeric(ii["item_latitude"], errors="coerce").astype(float).to_numpy()
    lon = pd.to_numeric(ii["item_longitude"], errors="coerce").astype(float).to_numpy()
    from geo import hav
    d = hav(c["clat"].to_numpy(), c["clon"].to_numpy(), lat, lon)
    f["dist_log"] = np.log1p(d).astype(np.float32)
    # «гео-PRF»: доля кандидатов топ-300 этого запроса в той же локации
    f["_iloc"] = iloc
    f["loc_in_cands"] = (f.groupby(["query_id", "_iloc"])["rank"].transform("size") /
                         f.groupby("query_id")["rank"].transform("size")).to_numpy(np.float32)

    # --- метаданные: фильтры, микрокатегория, PRF
    m = rm.filter_match(f[["query_id", "item_id", "score", "rank"]], queries, items)
    for col in ("q_vid", "q_tip", "m_vid", "m_tip", "m_all"):
        f[col] = m[col].to_numpy().astype(np.int8)
    proba = mc_model.predict_proba(queries)
    m = rm.microcat_prob(f[["query_id", "item_id", "score", "rank"]], queries, items, model=mc_model, proba=proba)
    f["p_mc"] = m["p_mc"].to_numpy(np.float32)
    f["mc_rank"] = np.log1p(m["mc_rank"].to_numpy()).astype(np.float32)
    pmax = dict(zip(queries["query_id"], proba.max(1)))
    f["p_mc_top"] = f["query_id"].map(pmax).to_numpy(np.float32)            # уверенность классификатора
    f["p_mc_rel"] = (f["p_mc"] / f["p_mc_top"].clip(lower=1e-6)).astype(np.float32)
    base = f[["query_id", "item_id", "rank"]]
    f["prf20"] = rm.prf_microcat(base, items, top=20).to_numpy(np.float32)
    f["prf50"] = rm.prf_microcat(base, items, top=50).to_numpy(np.float32)
    # ручной бонус из eda/11 как готовый признак (его же модель может и превзойти)
    f["manual"] = (f["score"] + 0.003 * f["m_vid"] + 0.0015 * f["m_tip"] + 0.007 * f["p_mc"]
                   + 0.002 * f["prf20"]).astype(np.float32)
    f["manual_pct"] = _in_query(f, "manual", "pct")
    f = f.drop(columns=["_iloc"])
    print(f"  признаки: {len(f)} строк, {len(FEATURES)} признаков за {time.time() - t0:.0f} с", flush=True)
    return f


FEATURES = [
    # geo2
    "g_rank", "g_score", "g_score_max", "g_gap50",
    # источники
    "bm25", "bm25_grank", "bm25_pct", "bm25_max", "bm25_zero", "cos", "cos_grank", "cos_pct", "cos_gap", "cos_z",
    # объявление
    "log_reviews", "rating", "has_rating", "price_stub", "log_price", "phone_hidden", "msg_forbidden",
    "desc_len", "title_len", "title_overlap",
    # запрос
    "q_words", "q_has_filter", "q_text_in_train", "q_delivery", "q_region", "q_russia",
    # гео
    "same_city", "loc_share", "loc_share_log", "dist_log", "loc_in_cands",
    # метаданные
    "q_vid", "q_tip", "m_vid", "m_tip", "m_all", "p_mc", "mc_rank", "p_mc_top", "p_mc_rel", "prf20", "prf50",
    "manual", "manual_pct",
]


SRC_FEATS = {"bm25", "bm25_grank", "bm25_pct", "bm25_max", "bm25_zero",
             "cos", "cos_grank", "cos_pct", "cos_gap", "cos_z"}
MODELS = ("hgb", "lgb_rank")     # lgb_bin / lgb_xendcg есть в make_model, но в итоговом прогоне не считаются (CPU)


# ====================================================================== модели и метрика

def recall50(f: pd.DataFrame, pred: np.ndarray, n_rel: pd.Series, k: int = 50) -> pd.Series:
    """Recall@k по запросам: сортировка кандидатов по pred (при равенстве — по исходному rank).
    n_rel — число правильных ответов запроса (включая не попавшие в кандидаты)."""
    q = f["query_id"].to_numpy()
    o = np.lexsort((f["rank"].to_numpy(), -pred, q))
    qs, ys = q[o], f["y"].to_numpy()[o]
    start = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    pos = np.arange(len(qs)) - np.repeat(start, np.diff(np.r_[start, len(qs)]))
    hit = pd.Series((ys * (pos < k)).astype(float), index=qs).groupby(level=0).sum()
    return (hit.reindex(n_rel.index).fillna(0) / n_rel).rename(f"recall@{k}")


def make_model(name: str):
    """Фиксированные заранее конфигурации (без подбора по валидации)."""
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              early_stopping=False, random_state=PY_SEED)
    import lightgbm as lgb
    common = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=PY_SEED, n_jobs=8, verbose=-1)
    if name == "lgb_rank":
        return lgb.LGBMRanker(objective="lambdarank", lambdarank_truncation_level=60, **common)
    if name == "lgb_xendcg":
        return lgb.LGBMRanker(objective="rank_xendcg", **common)
    if name == "lgb_bin":
        return lgb.LGBMClassifier(objective="binary", **common)
    raise ValueError(name)


def fit(name: str, f: pd.DataFrame, feats: list[str]):
    """Учит модель на кандидатах f (колонка y — метка). Для ранкеров группы = запросы (строки подряд)."""
    m = make_model(name)
    X, y = f[feats].to_numpy(np.float32), f["y"].to_numpy()
    if name.startswith("lgb_") and name != "lgb_bin":
        grp = f.groupby("query_id", sort=False).size().to_numpy()
        m.fit(X, y, group=grp)
    else:
        m.fit(X, y)
    return m


def predict(m, f: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Скор модели: вероятность класса 1 у классификаторов, сырой скор у ранкеров."""
    X = f[feats].to_numpy(np.float32)
    if hasattr(m, "predict_proba") and not type(m).__name__.endswith("Ranker"):
        return m.predict_proba(X)[:, 1]
    return m.predict(X)


def cv_predict(name: str, f: pd.DataFrame, feats: list[str], folds: np.ndarray, n_rel: pd.Series):
    """Вне-фолдовые предсказания; возвращает (oof, recall на обучающих частях по фолдам, модели)."""
    oof = np.zeros(len(f))
    fold_of_row = f["query_id"].map(folds).to_numpy()
    train_r, models = [], []
    for k in np.unique(fold_of_row):
        tr, te = fold_of_row != k, fold_of_row == k
        m = fit(name, f[tr], feats)
        oof[te] = predict(m, f[te], feats)
        ftr = f[tr]
        train_r.append(recall50(ftr, predict(m, ftr, feats), n_rel[n_rel.index.isin(ftr["query_id"])]).mean())
        models.append(m)
    return oof, train_r, models


def boot(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0):
    """Парный бутстреп разницы средних (a − b) по запросам: (разница, 2.5%, 97.5%)."""
    d = (a - b.reindex(a.index)).to_numpy()
    rng = np.random.default_rng(seed)
    mm = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


def to_cands(f: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Формат кандидатов: query_id, item_id (str), score (float32), rank (int32 с 1), топ-300."""
    out = pd.DataFrame({"query_id": f["query_id"].to_numpy(), "item_id": f["item_id"].astype(str).to_numpy(),
                        "score": pred.astype(np.float32), "old": f["rank"].to_numpy()})
    out = out.sort_values(["query_id", "score", "old"], ascending=[True, False, True], kind="mergesort")
    out["rank"] = (out.groupby("query_id", sort=False).cumcount() + 1).astype(np.int32)
    return out[["query_id", "item_id", "score", "rank"]].reset_index(drop=True)


# ====================================================================== эксперимент

def _load_items() -> pd.DataFrame:
    """Объявления корпуса с колонками для признаков (тексты, микрокатегория, гео, качество)."""
    cols = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text", "item_microcat_id",
            "item_location_id", "item_latitude", "item_longitude", "item_rating", "item_rating_reviews_count",
            "item_price", "item_is_phone_hidden", "item_is_message_forbidden"]
    it = pd.read_parquet(DATA / "benchmark_items.parquet", columns=cols)
    it["item_id"] = it["item_id"].astype(str)
    return it


def prepare_base(split, train_fit, items, cent, regions, queries) -> pd.DataFrame:
    """Признаки без источников (они считаются отдельно в cmd_sources) -> кеш work/ltr/feat_{split}.parquet.
    train_fit — train[mask] для val, полный train для bench."""
    path = LTR / f"feat_{split}.parquet"
    import rerank_meta as rm
    t = time.time()
    mc = rm.fit_microcat(train_fit)
    print(f"  {split}: классификатор микрокатегорий за {time.time() - t:.0f} с", flush=True)
    geo = GeoCtx(train_fit, cent, regions)
    c = pd.read_parquet(CANDS / f"hybrid_geo2_{split}.parquet")
    f = build_features(c, queries, items, None, mc, geo)
    f.to_parquet(path, index=False)
    return f


def cmd_run(base_only: bool = False):
    """LTR на 3000 запросах валидации перекрёстно (eda/16_ltr.md): базы, модели, абляции признаков, файлы."""
    from geo import centroids, load_geo_data
    from validation import add_flags, load_validation, region_location_ids, normalize_text
    t0 = time.time()
    vq, labels, mask = load_validation()
    vq["query_id"] = vq["query_id"].astype(str)
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    bq["query_id"] = bq["query_id"].astype(str)
    items = _load_items()
    regions = region_location_ids()
    need = not ((LTR / "feat_val.parquet").exists() and (LTR / "feat_bench.parquet").exists())
    if need:
        corpus, trg = load_geo_data()
        cent = centroids(corpus, trg)          # координаты объявлений (не ответы) — можно по всему train
        del corpus, trg
        tr = pd.read_parquet(DATA / "train.parquet", columns=["search_query", "search_infm_params_text",
                                                              "item_microcat_id", "search_location_id",
                                                              "item_location_id"])
        prepare_base("val", tr[mask], items, cent, regions, vq)
        tr_texts = set(normalize_text(tr["search_query"]))
        bq = add_flags(bq.assign(text_norm=normalize_text(bq["search_query"])), tr_texts, regions)
        prepare_base("bench", tr, items, cent, regions, bq)
        del tr
    if base_only:
        return
    fv = add_sources(pd.read_parquet(LTR / "feat_val.parquet"), pd.read_parquet(LTR / "src_val.parquet"))
    fb = add_sources(pd.read_parquet(LTR / "feat_bench.parquet"), pd.read_parquet(LTR / "src_bench.parquet"))
    print(f"признаки готовы за {time.time() - t0:.0f} с", flush=True)

    # метки и группы
    fv["y"] = [int(i in labels[q]) for q, i in zip(fv["query_id"], fv["item_id"])]
    n_rel = pd.Series({q: len(v) for q, v in labels.items()}).sort_index()
    fv = fv.sort_values(["query_id", "rank"], kind="mergesort").reset_index(drop=True)
    fb = fb.sort_values(["query_id", "rank"], kind="mergesort").reset_index(drop=True)

    # фолды: 2 половины, как A/B в rerank_meta_geo2 (seed 0), и 5 фолдов
    qids = np.array(sorted(labels))
    perm = np.random.default_rng(0).permutation(len(qids))
    fold2 = pd.Series(0, index=qids)
    fold2.iloc[perm[len(qids) // 2:]] = 1                      # 1 = половина B
    fold5 = pd.Series(np.arange(len(qids))[np.argsort(perm)] % 5, index=qids)

    # базы
    base = recall50(fv, -fv["rank"].to_numpy(float), n_rel)
    man_c = pd.read_parquet(CANDS / "hybrid_geo2_meta_val.parquet")
    man = pd.Series({q: len(set(g) & labels[q]) / len(labels[q])
                     for q, g in man_c[man_c["rank"] <= 50].groupby("query_id")["item_id"]}).reindex(n_rel.index).fillna(0)
    man_feat = recall50(fv, fv["manual"].to_numpy(), n_rel)
    isB = n_rel.index.isin(fold2[fold2 == 1].index)
    at300 = fv.groupby("query_id")["y"].sum().reindex(n_rel.index) / n_rel
    print(f"geo2: {base.mean():.4f} (A {base[~isB].mean():.4f}, B {base[isB].mean():.4f}); @300 {at300.mean():.4f}")
    print(f"geo2+бонусы (файл): {man.mean():.4f} (A {man[~isB].mean():.4f}, B {man[isB].mean():.4f}); "
          f"пересчёт по признаку manual: {man_feat.mean():.4f}")

    rows, oofs, models2 = [], {}, {}
    def report(tag, oof_r, train_r, feats):
        """Строка результатов варианта: OOF Recall@50, половины A/B, разница с ручными бонусами."""
        d = boot(oof_r, man)
        rows.append(dict(model=tag, n_feat=len(feats), oof=oof_r.mean(), A=oof_r[~isB].mean(), B=oof_r[isB].mean(),
                         train=np.mean(train_r), d_man=d[0], lo=d[1], hi=d[2]))
        print(f"{tag:28s} OOF {oof_r.mean():.4f} (A {oof_r[~isB].mean():.4f} B {oof_r[isB].mean():.4f}) "
              f"train {np.mean(train_r):.4f}  Δ к бонусам {d[0]:+.4f} [{d[1]:+.4f}; {d[2]:+.4f}]  "
              f"{time.time() - t0:.0f} с", flush=True)

    for name in MODELS:
        oof, tr_r, ms = cv_predict(name, fv, FEATURES, fold2, n_rel)
        r = recall50(fv, oof, n_rel)
        oofs[(name, 2)], models2[name] = r, ms
        report(f"{name} (2 фолда)", r, tr_r, FEATURES)
        np.save(LTR / f"oof2_{name}.npy", oof)

    # абляции (lgb_rank, 2 фолда): без готовой ручной формулы; без скоров источников
    for tag, drop in (("без manual", {"manual", "manual_pct"}), ("без источников", SRC_FEATS)):
        fs = [c for c in FEATURES if c not in drop]
        oof, tr_r, _ = cv_predict("lgb_rank", fv, fs, fold2, n_rel)
        report(f"lgb_rank {tag} (2 ф.)", recall50(fv, oof, n_rel), tr_r, fs)

    # лучшая по 2 фолдам модель -> 5 фолдов
    best = max(MODELS, key=lambda n: oofs[(n, 2)].mean())
    for name in [best]:
        oof, tr_r, _ = cv_predict(name, fv, FEATURES, fold5, n_rel)
        r = recall50(fv, oof, n_rel)
        oofs[(name, 5)] = r
        report(f"{name} (5 фолдов)", r, tr_r, FEATURES)
        np.save(LTR / f"oof5_{name}.npy", oof)

    res = pd.DataFrame(rows)
    res.to_csv(LTR / "results.csv", index=False)

    # срезы для лучшей модели (2 фолда)
    rb = oofs[(best, 2)]
    d = vq.set_index("query_id").loc[n_rel.index]
    sl = {"все": np.ones(len(d), bool), "с фильтром": d.has_filter.to_numpy(), "без фильтра": ~d.has_filter.to_numpy(),
          "текста нет в train": ~d.text_in_train.to_numpy(), "ответ без истории": ~d.target_has_history.to_numpy(),
          "у ответа есть история": d.target_has_history.to_numpy(),
          "поиск по городу": ~d.is_region.to_numpy(), "регион или вся Россия": d.is_region.to_numpy()}
    srows = []
    for k, mk in sl.items():
        dd = boot(rb[mk], man[mk])
        srows.append(dict(срез=k, n=int(mk.sum()), geo2=base[mk].mean(), бонусы=man[mk].mean(), ltr=rb[mk].mean(),
                          d=dd[0], lo=dd[1], hi=dd[2]))
    sdf = pd.DataFrame(srows)
    sdf.to_csv(LTR / "slices.csv", index=False)
    print(sdf.round(4).to_string(index=False))

    # важность: gain (lightgbm) и перестановочная по Recall@50 на вне-фолдовых данных
    imp = pd.DataFrame(index=FEATURES)
    lg = models2["lgb_rank"]
    imp["gain_lgb_rank"] = np.mean([m.booster_.feature_importance("gain") for m in lg], axis=0)
    imp["gain_lgb_rank"] /= imp["gain_lgb_rank"].sum()
    ms = models2[best]
    fold_of_row = fv["query_id"].map(fold2).to_numpy()
    rng = np.random.default_rng(0)
    ref = oofs[(best, 2)].mean()
    drops = {}
    # перестановочная важность — для топ-20 по gain и всех признаков источников (иначе долго на CPU)
    for col in list(dict.fromkeys(imp["gain_lgb_rank"].sort_values(ascending=False).index[:20].tolist()
                                  + sorted(SRC_FEATS))):
        g = fv
        saved = g[col].to_numpy().copy()
        g[col] = saved[rng.permutation(len(saved))]
        oof = np.zeros(len(g))
        for k in (0, 1):
            te = fold_of_row == k
            oof[te] = predict(ms[k], g[te], FEATURES)
        g[col] = saved
        drops[col] = ref - recall50(g, oof, n_rel).mean()
    imp[f"perm_{best}"] = pd.Series(drops)
    imp = imp.sort_values(f"perm_{best}", ascending=False)
    imp.to_csv(LTR / "importance.csv")
    print(imp.round(4).head(20).to_string())

    # запись: val — вне-фолдовые скоры лучшей модели (2 фолда), bench — модель на всех 3000
    oof_best = np.load(LTR / f"oof2_{best}.npy")
    cv_ = to_cands(fv, oof_best)
    cv_.to_parquet(CANDS / "hybrid_geo2_ltr_val.parquet", index=False)
    m_all = fit(best, fv, FEATURES)
    pb = predict(m_all, fb, FEATURES)
    cb = to_cands(fb, pb)
    assert cb.groupby("query_id").size().eq(300).all() and cb["query_id"].nunique() == len(bq)
    cb.to_parquet(CANDS / "hybrid_geo2_ltr_bench.parquet", index=False)
    g0 = pd.read_parquet(CANDS / "hybrid_geo2_bench.parquet")
    gm = pd.read_parquet(CANDS / "hybrid_geo2_meta_bench.parquet")
    top = cb[cb["rank"] <= 50]
    for nm, o in (("geo2", g0), ("geo2+бонусы", gm)):
        same = top.merge(o[o["rank"] <= 50], on=["query_id", "item_id"]).shape[0] / len(top)
        print(f"бенчмарк: топ-50 LTR совпадает с {nm} на {same:.3f}")
    # проверка файла валидации
    chk = pd.Series({q: len(set(g) & labels[q]) / len(labels[q])
                     for q, g in cv_[cv_["rank"] <= 50].groupby("query_id")["item_id"]}).reindex(n_rel.index).fillna(0)
    print(f"лучшая модель {best}; файл val Recall@50 = {chk.mean():.4f}; всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sources", "base", "run"])
    a = ap.parse_args()
    if a.cmd == "sources":
        cmd_sources()
    else:
        cmd_run(base_only=a.cmd == "base")   # base — только признаки без источников (можно параллельно)
