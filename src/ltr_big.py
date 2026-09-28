"""LTR на большой обучающей выборке из train (см. eda/18_ltr_big.md).

Идея: вместо 3000 размеченных запросов валидации учить LightGBM lambdarank (признаки src/ltr.py)
на ~21 тыс. запросов из train[train_mask], у которых выбранное объявление есть в корпусе.
Валидация (3000) остаётся ЧИСТОЙ отложенной проверкой: на её запросах модель не учится,
тексты валидации из обучающего пула исключены.

Шаги (запуск из корня репозитория):
    python src/ltr_big.py sample     # обучающий пул, фолды по тексту, какие строки train убрать для фолда
    python src/ltr_big.py retrieve   # bge-m3 запросов (GPU ≤ 2 ГБ), кандидаты geo2 топ-300 + скоры источников
    python src/ltr_big.py features   # признаки build_features (+ свои) по частям -> work/ltr_big/feat_*.parquet
    python src/ltr_big.py leakdiag   # проверка утечки: сопоставимые слои пула и валидации, «протёкший» классификатор
    python src/ltr_big.py train      # кривая обучения 2k/4k/8k/все и варианты (вес истории, без качества, ...)
    python src/ltr_big.py final      # ансамбль 5 моделей -> work/cands/hybrid_geo2_ltrbig_{val,bench}.parquet
    # дообученный эмбеддер (обучен на train[mask] -> видел строки запросов пула, в признаки LTR не идёт):
    python src/ltr_big.py ftcands --ft-name e5-small-ft   # список «BM25 + bge-m3 + ft» (RRF 1/0.5/0.5), val/bench
    python src/ltr_big.py fuse    --ft-name e5-small-ft   # (а) позднее слияние рангов, вес на половине A
    python src/ltr_big.py stack   --ft-name e5-small-ft   # (в) стекер на 3000 валидации перекрёстно
                                                          # -> work/ltr_big/stack_{val,bench}.parquet
    (итоговый hybrid_geo2_ltrbig_ft_{val,bench}.parquet = стекер e5-small-ft, см. eda/18_ltr_big.md)

Схема против утечки (главное):
    всё, что для запроса q строится по train (классификатор микрокатегорий, гео-карты GeoMaps,
    доли пар локаций GeoCtx, флаг «текст есть в train»), строится БЕЗ строк самого q.
    Пул делится на K=5 фолдов по нормализованному тексту; для фолда f объекты учатся на
        train[mask] без строк всех запросов пула из фолда f
                    и без ВСЕХ строк текстов фолда f, назначенных «новыми» (режим unseen, см. ниже).
    Для валидации — полный train[mask] (как в src/ltr.py), для бенчмарка — полный train.
    Режим unseen: в бенчмарке 63% текстов не встречаются в train. У пула таких мало (валидация забрала
    большую часть редких текстов), поэтому часть текстов пула делаем «новыми», убирая все их строки
    из train фолда. Убрать больше строк — всегда безопасно с точки зрения утечки.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

ROOT = SRC.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
CANDS = WORK / "cands"
OUT = WORK / "ltr_big"
K_FOLDS = 5
SEED = 0
TRAIN_SETS = [f"tr{f}" for f in range(K_FOLDS)]


# ====================================================================== общие загрузки

def load_train_full() -> pd.DataFrame:
    """train в исходном порядке строк: поисковые колонки, ответ, координаты, query_key/text_norm."""
    from validation import add_query_key
    cols = ["search_query", "search_location_id", "search_infm_params_text", "search_is_delivery_search",
            "search_category", "item_id", "item_location_id", "item_latitude", "item_longitude", "item_microcat_id"]
    tr = pd.read_parquet(DATA / "train.parquet", columns=cols)
    tr["item_id"] = tr["item_id"].astype(str)
    tr["lat"] = pd.to_numeric(tr["item_latitude"], errors="coerce").astype(float)
    tr["lon"] = pd.to_numeric(tr["item_longitude"], errors="coerce").astype(float)
    tr = tr.drop(columns=["item_latitude", "item_longitude"])
    return add_query_key(tr)


def train_fit_for(split: str, tr: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Строки train, по которым строятся объекты для набора запросов split (см. docstring модуля)."""
    if split == "bench":
        return tr
    if split == "val":
        return tr[mask]
    f = int(split[2:])
    excl = np.load(OUT / "excl_fold.npy")
    return tr[mask & (excl != f)]


# ====================================================================== шаг 1: пул

def cmd_sample():
    """Пул обучающих запросов: запросы train[mask] с ответом в корпусе, кроме текстов валидации.

    Пишет pool_queries.parquet (query_id tr#####, поля запроса, fold, mode), pool_labels.parquet,
    excl_fold.npy (для строки train — номер фолда, из train которого она убрана, иначе −1).
    """
    from validation import check_mask, has_filter, load_validation, add_flags, region_location_ids
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_train_full()
    vq, vlabels, mask = load_validation()
    check_mask(tr, mask)
    corpus = set(pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])["item_id"].astype(str))
    tm = tr[mask]
    inc = tm["item_id"].isin(corpus)
    val_texts = set(vq["text_norm"])
    el = tm[inc & ~tm["text_norm"].isin(val_texts)]
    labels = el.groupby("query_key")["item_id"].agg(lambda s: sorted(set(s)))
    q = el.drop_duplicates("query_key").drop(columns=["item_id", "item_location_id", "lat", "lon",
                                                        "item_microcat_id"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    q = q.iloc[rng.permutation(len(q))].reset_index(drop=True)
    q.insert(0, "query_id", [f"tr{i:05d}" for i in range(len(q))])

    # фолды по тексту
    texts = q["text_norm"].drop_duplicates().to_numpy()
    fold_of_text = dict(zip(texts, rng.permutation(len(texts)) % K_FOLDS))
    q["fold"] = q["text_norm"].map(fold_of_text).astype(int)
    # сколько у текста запросов в train[mask] вне пула: если 0 — текст «новый» сам собой
    n_all = tm.groupby("text_norm")["query_key"].nunique()
    n_pool = q.groupby("text_norm").size()
    other = (n_all.reindex(n_pool.index) - n_pool).astype(int)
    # режим unseen: доля новых текстов в бенчмарке 0.626; добираем её редкими текстами
    # (меньше «других» запросов = правдоподобнее, что текст мог не встретиться), тай-брейк случайный
    target_unseen = 0.60
    t = pd.DataFrame({"other": other, "r": rng.random(len(other))})
    t["mode"] = np.where(t["other"] == 0, "unseen_nat", "seen")
    need = int(target_unseen * len(t)) - int((t["mode"] == "unseen_nat").sum())
    cand = t[t["mode"] == "seen"].sort_values(["other", "r"])
    if need > 0:
        t.loc[cand.index[:need], "mode"] = "unseen_forced"
    q["mode"] = q["text_norm"].map(t["mode"])
    print(f"пул: {len(q)} запросов, {len(texts)} текстов; режимы текстов:\n{t['mode'].value_counts().to_string()}")
    print(f"  граница «other» для unseen_forced: ≤ {cand['other'].iloc[max(need - 1, 0)]}")

    # какие строки train[mask] убрать из train фолда f
    excl = np.full(len(tr), -1, np.int8)
    key_fold = dict(zip(q["query_key"], q["fold"]))
    kf = tr["query_key"].map(key_fold)
    forced = {tx: fold_of_text[tx] for tx in t.index[t["mode"] == "unseen_forced"]}
    tf = tr["text_norm"].map(forced)
    fold_row = kf.fillna(tf)
    ok = fold_row.notna().to_numpy() & mask
    excl[ok] = fold_row[ok].astype(int).to_numpy()
    np.save(OUT / "excl_fold.npy", excl)

    # флаги относительно train своего фолда + история ответа
    regions = region_location_ids()
    parts = []
    for f in range(K_FOLDS):
        fit = tr[mask & (excl != f)]
        qf = add_flags(q[q["fold"] == f], set(fit["text_norm"]), regions)
        items_f = set(fit["item_id"])
        qf["target_has_history"] = [bool(set(labels[k]) & items_f) for k in qf["query_key"]]
        # проверка: ни одной строки своих запросов в train фолда
        assert not fit["query_key"].isin(set(qf["query_key"])).any()
        parts.append(qf)
    q = pd.concat(parts).sort_values("query_id").reset_index(drop=True)
    assert (q.loc[q["mode"] != "seen", "text_in_train"] == False).all()  # noqa: E712
    q["n_text"] = q["text_norm"].map(n_pool).astype(int)
    q.to_parquet(OUT / "pool_queries.parquet", index=False)
    pd.DataFrame({"query_id": q["query_id"], "item_ids": q["query_key"].map(labels)}) \
        .to_parquet(OUT / "pool_labels.parquet", index=False)
    print(q.groupby(["text_in_train", "has_filter"]).size().rename("запросов").to_string())
    print(f"у ответа есть история: {q['target_has_history'].mean():.3f}; регион: {q['is_region'].mean():.3f}; "
          f"строк убрано по фолдам: {np.bincount(excl[excl >= 0])}")


def load_pool():
    """Обучающий пул LTR: запросы (с фолдом и режимом) и их правильные ответы."""
    q = pd.read_parquet(OUT / "pool_queries.parquet")
    lab = pd.read_parquet(OUT / "pool_labels.parquet")
    labels = {a: set(b) for a, b in zip(lab["query_id"], lab["item_ids"])}
    return q, labels


def queries_of(split: str) -> pd.DataFrame:
    """Запросы набора с флагом text_in_train относительно его train_fit."""
    from validation import add_flags, load_validation, normalize_text, region_location_ids
    if split == "val":
        q = load_validation()[0]
    elif split == "bench":
        q = pd.read_parquet(DATA / "benchmark_queries.parquet")
        tr_texts = set(normalize_text(pd.read_parquet(DATA / "train.parquet", columns=["search_query"])["search_query"]))
        q = add_flags(q, tr_texts, region_location_ids())
    else:
        q = load_pool()[0]
        q = q[q["fold"] == int(split[2:])]
    q = q.reset_index(drop=True)
    q["query_id"] = q["query_id"].astype(str)
    return q


# ====================================================================== шаг 2: кандидаты + источники

def cmd_retrieve(splits: list[str]):
    """bge-m3 запросов на GPU (fp16, VRAM ≤ 2 ГБ) и кандидаты geo2 топ-300 тем же кодом, что pipeline.py,
    плюс скоры источников для признаков (BM25 и косинус, их глобальные ранги в корпусе)."""
    import torch
    from pipeline import (Config, build_geo_maps, build_sparse_index, encode_query_embeddings,
                          load_corpus_embeddings, geo_rule)
    from geo import tiered_top, rrf
    from sparse import SparseIndex
    from validation import region_location_ids
    cfg = Config()
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        torch.cuda.set_per_process_memory_fraction(min(1.0, 2.0 / total))
    t0 = time.time()
    items = pd.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id", "item_location_id", "item_latitude", "item_longitude",
                                     "item_title_raw"])
    items["item_id"] = items["item_id"].astype(str)
    items["lat"] = pd.to_numeric(items["item_latitude"], errors="coerce").astype(float)
    items["lon"] = pd.to_numeric(items["item_longitude"], errors="coerce").astype(float)
    item_ids = items["item_id"].to_numpy()
    tr = load_train_full()
    from validation import load_validation
    mask = load_validation()[2]
    regions = region_location_ids()

    # 1) эмбеддинги запросов всех наборов одним прогоном модели
    qs = {s: queries_of(s) for s in splits}
    todo = [s for s in splits if not (OUT / f"qemb_{s}.npy").exists()]
    if todo:
        allq = pd.concat([qs[s] for s in todo], ignore_index=True)
        e = encode_query_embeddings(allq, cfg)
        o = 0
        for s in todo:
            np.save(OUT / f"qemb_{s}.npy", e[o:o + len(qs[s])])
            o += len(qs[s])
        print(f"bge-m3: {len(allq)} запросов за {time.time() - t0:.0f} с, "
              f"пик VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} ГБ", flush=True)
        del e
    sparse = build_sparse_index(items, cfg)
    emb = load_corpus_embeddings(items, cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = torch.from_numpy(emb).to(dev, torch.float16 if dev == "cuda" else torch.float32)
    del emb
    N, K, bs = len(item_ids), cfg.n_cands, cfg.dense_batch
    all_pos = np.arange(N)

    for s in splits:
        path = OUT / f"cands_{s}.parquet"
        if path.exists():
            continue
        q, q_emb = qs[s], np.load(OUT / f"qemb_{s}.npy")
        maps = build_geo_maps(items, tr, train_fit_for(s, tr, mask), regions)
        locs, texts = q["search_location_id"].to_numpy(), q["search_query"].to_numpy()
        pos_out, sc_out, bm, bmr, cs, csr = [], [], [], [], [], []
        t1 = time.time()
        for b0 in range(0, len(q), bs):
            with torch.no_grad():
                dsc = (torch.from_numpy(q_emb[b0:b0 + bs]).to(dev, D.dtype) @ D.T).float().cpu().numpy()
            for j, dv in enumerate(dsc):
                i = b0 + j
                loc = int(locs[i])
                ssc = sparse.scores(texts[i])
                # --- ровно как pipeline.retrieve_candidates
                g_s = SparseIndex._top(all_pos, ssc, 2 * K)
                g_d = np.argpartition(-dv, 2 * K)[:2 * K]
                g_d = g_d[np.argsort(-dv[g_d], kind="stable")]
                rule = geo_rule(loc, maps.regions, cfg)
                pool = maps.tiers(maps.base_rule(rule), loc)
                top_sparse, _ = tiered_top(ssc, pool, K, g_s, positive=True)
                top_dense, _ = tiered_top(dv, pool, K, g_d, positive=False)
                bonus_mask, bonus_w = maps.bonus_mask(rule, loc)
                fused, fsc = rrf([top_sparse, top_dense], k_rrf=cfg.rrf_k, n_out=K, bonus=bonus_mask, bonus_w=bonus_w)
                pos_out.append(fused)
                sc_out.append(fsc)
                # --- скоры источников: значение и глобальный ранг (1 + число объявлений со скором выше)
                v = dv[fused]
                cs.append(v)
                csr.append(N - np.searchsorted(np.sort(dv), v, side="right") + 1)
                b = ssc[fused]
                ps = np.sort(ssc[ssc > 0])
                bm.append(b)
                bmr.append(len(ps) - np.searchsorted(ps, b, side="right") + 1)
            if (b0 // bs) % 20 == 0:
                print(f"  {s}: {min(b0 + bs, len(q))}/{len(q)} за {time.time() - t1:.0f} с", flush=True)
        n = [len(p) for p in pos_out]
        assert (np.array(n) == K).all()
        c = pd.DataFrame({"query_id": np.repeat(q["query_id"].to_numpy(), n),
                          "item_id": item_ids[np.concatenate(pos_out)].astype(str),
                          "score": np.concatenate(sc_out).astype(np.float32),
                          "rank": np.tile(np.arange(1, K + 1), len(q)).astype(np.int32),
                          "bm25": np.concatenate(bm).astype(np.float32),
                          "bm25_grank": np.concatenate(bmr).astype(np.int32),
                          "cos": np.concatenate(cs).astype(np.float32),
                          "cos_grank": np.concatenate(csr).astype(np.int32)})
        c.to_parquet(path, index=False)
        print(f"{s}: {len(q)} запросов, {len(c)} строк за {time.time() - t1:.0f} с", flush=True)
        # сверка с эталонными кандидатами geo2 (pipeline / geo_run)
        ref = CANDS / f"hybrid_geo2_{s}.parquet"
        if ref.exists():
            r = pd.read_parquet(ref)
            a = c[c["rank"] <= 50].groupby("query_id")["item_id"].agg(frozenset)
            bb = r[r["rank"] <= 50].groupby("query_id")["item_id"].agg(frozenset)
            print(f"  сверка с {ref.name}: топ-50 совпадает у {(a == bb.reindex(a.index)).mean():.4f} запросов")
        del maps
        gc.collect()


# ====================================================================== шаг 3: признаки

def _norm(s: pd.Series) -> pd.Series:
    """Нижний регистр, ё -> е, схлопнутые пробелы."""
    return s.fillna("").astype(str).str.lower().str.replace("ё", "е", regex=False) \
        .str.replace(r"\s+", " ", regex=True).str.strip()


def extra_features(f: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Свои добавки к build_features: точное вхождение фразы запроса в заголовок и в текст объявления
    (заголовок + параметры + начало описания), доля слов запроса в тексте объявления,
    цена относительно медианы кандидатов запроса."""
    from ltr import _tokens
    qtext = dict(zip(queries["query_id"], _norm(queries["search_query"])))
    it = items.set_index("item_id")
    title = _norm(it["item_title_raw"])
    body = title + " " + _norm(it["item_infm_params_text"]) + " " + _norm(it["item_description_raw"].str[:500])
    ti = title.reindex(f["item_id"]).to_numpy()
    bo = body.reindex(f["item_id"]).to_numpy()
    qs = f["query_id"].map(qtext).to_numpy()
    f["phrase_title"] = np.fromiter((len(a) > 0 and a in b for a, b in zip(qs, ti)), bool, len(f)).astype(np.int8)
    f["phrase_text"] = np.fromiter((len(a) > 0 and a in b for a, b in zip(qs, bo)), bool, len(f)).astype(np.int8)
    qtok = {q: set(_tokens(s)) for q, s in qtext.items()}
    need = pd.unique(f["item_id"])                     # токены только нужных объявлений (память)
    btok = {i: set(_tokens(s)) for i, s in zip(need, body.reindex(need).to_numpy())}
    ov = np.zeros(len(f), np.float32)
    for k, (q, i) in enumerate(zip(f["query_id"].to_numpy(), f["item_id"].to_numpy())):
        a = qtok.get(q)
        if a:
            ov[k] = len(a & btok.get(i, set())) / len(a)
    f["text_overlap"] = ov
    med = f.groupby("query_id", sort=False)["log_price"].transform("median")
    f["price_rel"] = (f["log_price"] - med).astype(np.float32)
    return f


EXTRA = ["phrase_title", "phrase_text", "text_overlap", "price_rel"]


def cmd_features(splits: list[str]):
    """Признаки по наборам: объекты (классификатор, GeoCtx) — по train_fit набора. Хранится float32."""
    import rerank_meta as rm
    from ltr import FEATURES, GeoCtx, _load_items, build_features
    from geo import centroids
    from validation import load_validation, region_location_ids
    t0 = time.time()
    items = _load_items()
    tr = load_train_full()
    vq, vlabels, mask = load_validation()
    regions = region_location_ids()
    corpus = items[["item_id", "item_location_id"]].copy()
    corpus["lat"] = pd.to_numeric(items["item_latitude"], errors="coerce").astype(float)
    corpus["lon"] = pd.to_numeric(items["item_longitude"], errors="coerce").astype(float)
    cent = centroids(corpus, tr)
    # дальше нужны только колонки для классификатора и гео-долей (память: пик — fit_microcat)
    tr = tr[["search_query", "search_infm_params_text", "item_microcat_id", "search_location_id",
             "item_location_id"]].copy()
    gc.collect()
    pool_labels = load_pool()[1]
    for s in splits:
        path = OUT / f"feat_{s}.parquet"
        if path.exists():
            continue
        t1 = time.time()
        fit = train_fit_for(s, tr, mask)
        q = queries_of(s)
        mc = rm.fit_microcat(fit)
        geo = GeoCtx(fit, cent, regions)
        call = pd.read_parquet(OUT / f"cands_{s}.parquet")
        labels = vlabels if s == "val" else (pool_labels if s.startswith("tr") else None)
        keep = ["query_id", "item_id", "rank"] + FEATURES + EXTRA + (["y"] if labels is not None else [])
        parts = []
        # по частям по 1000 запросов: все признаки внутри-запросные, объекты (mc, geo) общие для набора
        for c0 in range(0, len(q), 1000):
            qc = q.iloc[c0:c0 + 1000]
            c = call[call["query_id"].isin(set(qc["query_id"]))]
            src = c[["query_id", "item_id", "bm25", "bm25_grank", "cos", "cos_grank"]]
            f = build_features(c[["query_id", "item_id", "score", "rank"]], qc, items, src, mc, geo)
            f = extra_features(f, qc, items)
            if labels is not None:
                f["y"] = np.fromiter((i in labels[qq] for qq, i in zip(f["query_id"], f["item_id"])), bool,
                                     len(f)).astype(np.int8)
            f = f[keep].copy()
            for col in FEATURES + EXTRA:
                f[col] = f[col].astype(np.float32)
            parts.append(f)
            del c, src
            gc.collect()
        f = pd.concat(parts, ignore_index=True)
        del parts, call
        f.to_parquet(path, index=False)
        print(f"{s}: {len(f)} строк, признаки за {time.time() - t1:.0f} с (всего {time.time() - t0:.0f} с)",
              flush=True)
        del f, mc, geo, fit
        gc.collect()


# ====================================================================== шаг 4: модели

FEATS_ALL = None          # заполняется в load_feats: FEATURES из ltr.py + EXTRA
QUALITY = ["log_reviews", "rating", "has_rating", "desc_len"]   # признаки, работающие как популярность


def load_feats(split: str) -> pd.DataFrame:
    """Признаки набора из work/ltr_big/feat_<split>.parquet; заодно заполняет FEATS_ALL."""
    from ltr import FEATURES
    global FEATS_ALL
    FEATS_ALL = FEATURES + EXTRA
    f = pd.read_parquet(OUT / f"feat_{split}.parquet")
    return f


def lgb_params(seed: int = 0) -> dict:
    """Заданы заранее (не подбирались по валидации). Больше данных -> больше листьев, чем в ltr.py."""
    return dict(objective="lambdarank", lambdarank_truncation_level=60, n_estimators=3000, learning_rate=0.05,
                num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=seed, n_jobs=8, verbose=-1)


def groups(qid: np.ndarray) -> np.ndarray:
    """Размеры групп для строк, где запросы идут подряд."""
    start = np.r_[0, np.flatnonzero(qid[1:] != qid[:-1]) + 1]
    return np.diff(np.r_[start, len(qid)])


def fit_ranker(tr: pd.DataFrame, es: pd.DataFrame, feats: list[str], w_q: pd.Series | None, seed: int = 0):
    """LGBMRanker с ранней остановкой по NDCG@50 на отдельной части ОБУЧАЮЩИХ запросов (не валидации).
    w_q — вес запроса (одинаковый для всех его строк)."""
    import lightgbm as lgb
    m = lgb.LGBMRanker(**lgb_params(seed))
    q_tr = tr["query_id"].to_numpy()
    sw = None if w_q is None else tr["query_id"].map(w_q).to_numpy(np.float32)
    m.fit(tr[feats].to_numpy(np.float32), tr["y"].to_numpy(), group=groups(q_tr), sample_weight=sw,
          eval_set=[(es[feats].to_numpy(np.float32), es["y"].to_numpy())], eval_group=[groups(es["query_id"].to_numpy())],
          eval_at=[50], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def recall_of(f: pd.DataFrame, pred: np.ndarray, labels: dict) -> pd.Series:
    """Recall@50 по запросам при сортировке кандидатов f по pred (правильные вне кандидатов тоже в знаменателе)."""
    from ltr import recall50
    qs = f["query_id"].unique()
    n_rel = pd.Series({q: len(labels[q]) for q in qs}).sort_index()
    return recall50(f, pred, n_rel)


def cands_recall(path: Path, labels: dict) -> pd.Series:
    """Recall@50 файла кандидатов (query_id, item_id, rank) по запросам labels."""
    c = pd.read_parquet(path)
    top = c[c["rank"] <= 50].groupby("query_id")["item_id"].agg(set)
    return pd.Series({q: len(top.get(q, set()) & v) / len(v) for q, v in labels.items()}).sort_index()


def boot(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0):
    """Парный бутстреп разницы средних (a − b) по запросам: (разница, 2.5%, 97.5%)."""
    d = (a - b.reindex(a.index)).to_numpy()
    rng = np.random.default_rng(seed)
    mm = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


def pool_weights(pq: pd.DataFrame, bench_q: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    """Вес запроса пула: (1 / n_text^alpha) × поправка слоя (текст в train × фильтр) до долей бенчмарка.
    n_text — число запросов того же текста в используемой выборке."""
    n_text = pq.groupby("text_norm")["query_id"].transform("size")
    base = 1.0 / n_text.to_numpy() ** alpha
    st = list(zip(pq["text_in_train"], pq["has_filter"]))
    mass = pd.Series(base).groupby(pd.Series(st)).sum() / base.sum()
    target = bench_q[["text_in_train", "has_filter"]].value_counts(normalize=True)
    corr = {k: target[k] / mass[k] for k in mass.index}
    w = base * np.array([corr[k] for k in st])
    return pd.Series(w / w.mean(), index=pq["query_id"].to_numpy())


def cmd_train():
    """Кривая обучения LTR-big (2k/4k/8k/все запросы пула) и варианты против опоры на популярность."""
    from validation import load_validation
    from ltr import to_cands
    t0 = time.time()
    RES = OUT / "res"
    RES.mkdir(exist_ok=True)
    vq, vlabels, _ = load_validation()
    pq, plabels = load_pool()
    bq = queries_of("bench")
    fv = load_feats("val")
    feats = FEATS_ALL
    cols = ["query_id", "rank", "y"] + feats
    ft = pd.concat([pd.read_parquet(OUT / f"feat_{s}.parquet", columns=cols) for s in TRAIN_SETS],
                   ignore_index=True)
    print(f"пул: {len(ft)} строк, {ft['query_id'].nunique()} запросов; {time.time() - t0:.0f} с", flush=True)
    # запросы пула без правильного ответа в топ-300 не дают градиента lambdarank — убираем
    has_pos = ft.groupby("query_id", sort=False)["y"].transform("max").to_numpy() > 0
    print(f"  с ответом в топ-300: {ft.loc[has_pos, 'query_id'].nunique() / ft['query_id'].nunique():.3f}")

    # ---------------- базы на валидации
    n_rel = pd.Series({q: len(v) for q, v in vlabels.items()}).sort_index()
    r_geo2 = recall_of(fv, -fv["rank"].to_numpy(float), vlabels)
    r_man = cands_recall(CANDS / "hybrid_geo2_meta_val.parquet", vlabels)
    r_ltr3k = cands_recall(CANDS / "hybrid_geo2_ltr_val.parquet", vlabels)
    old = pd.read_parquet(WORK / "ltr" / "feat_val.parquet", columns=["query_id", "item_id", "rank"])
    old = old.sort_values(["query_id", "rank"], kind="mergesort").reset_index(drop=True)
    old["y"] = [int(i in vlabels[q]) for q, i in zip(old["query_id"], old["item_id"])]
    r_ltr3k5 = recall_of(old, np.load(WORK / "ltr" / "oof5_lgb_rank.npy"), vlabels)
    print(f"валидация: geo2 (мои кандидаты) {r_geo2.mean():.4f}; geo2+бонусы {r_man.mean():.4f}; "
          f"LTR-3k 2 фолда {r_ltr3k.mean():.4f}; 5 фолдов {r_ltr3k5.mean():.4f}", flush=True)

    # ---------------- проверка утечки: распределения у правильных ответов, пул против валидации
    leak_check(ft, fv, pq, vq, plabels, vlabels, RES)

    # ---------------- разбиение пула для ранней остановки: 10% текстов (по тексту, как фолды)
    rng = np.random.default_rng(1)
    texts = pq["text_norm"].drop_duplicates().to_numpy()
    es_texts = set(rng.choice(texts, size=len(texts) // 10, replace=False))
    pq["es"] = pq["text_norm"].isin(es_texts)
    es_ids = set(pq.loc[pq["es"], "query_id"])
    is_es = ft["query_id"].isin(es_ids).to_numpy()
    es = ft[is_es & has_pos]
    fit_pool = pq[~pq["es"]]
    # вложенные подвыборки для кривой: по одному запросу на текст, случайный порядок текстов
    one = fit_pool.sample(frac=1.0, random_state=2).drop_duplicates("text_norm")
    subsets = {"2k": one.head(2000), "4k": one.head(4000), "8k": one.head(8000),
               f"все ({len(fit_pool)})": fit_pool}

    sl_q = vq.set_index("query_id").loc[n_rel.index]
    SL = {"все": np.ones(len(sl_q), bool), "с фильтром": sl_q.has_filter.to_numpy(),
          "без фильтра": ~sl_q.has_filter.to_numpy(), "текст есть в train": sl_q.text_in_train.to_numpy(),
          "текста нет в train": ~sl_q.text_in_train.to_numpy(), "поиск по городу": ~sl_q.is_region.to_numpy(),
          "регион или вся Россия": sl_q.is_region.to_numpy(), "у ответа есть история": sl_q.target_has_history.to_numpy(),
          "ответ без истории": ~sl_q.target_has_history.to_numpy()}
    rows, preds = [], {}

    def run(tag, sub, w_mode="strata", feats_=None, seed=0, hist_w=1.0):
        """Обучить вариант на подвыборке пула, оценить на валидации и записать строку результатов."""
        feats_ = feats_ or feats
        ids = set(sub["query_id"])
        tr_ = ft[ft["query_id"].isin(ids).to_numpy() & has_pos]
        w = pool_weights(sub, bq) if w_mode == "strata" else pd.Series(1.0, index=sub["query_id"].to_numpy())
        if hist_w != 1.0:
            h = sub.set_index("query_id")["target_has_history"].reindex(w.index).to_numpy()
            w = w * np.where(h, hist_w, 1.0)
            w = w / w.mean()
        t1 = time.time()
        m = fit_ranker(tr_, es, feats_, w, seed)
        p = m.predict(fv[feats_].to_numpy(np.float32))
        r = recall_of(fv, p, vlabels)
        r_es = recall_of(es, m.predict(es[feats_].to_numpy(np.float32)), plabels)
        d_man, d_3k = boot(r, r_man), boot(r, r_ltr3k)
        row = dict(вариант=tag, запросов=len(ids), деревьев=m.best_iteration_, val=r.mean(), es_пула=r_es.mean(),
                   d_man=d_man[0], man_lo=d_man[1], man_hi=d_man[2], d_3k=d_3k[0], k3_lo=d_3k[1], k3_hi=d_3k[2])
        for k, mk in SL.items():
            row["R " + k] = r[mk].mean()
        dn = boot(r[SL["ответ без истории"]], r_man[SL["ответ без истории"]])
        row.update(nohist_d_man=dn[0], nohist_lo=dn[1], nohist_hi=dn[2])
        dn3 = boot(r[SL["ответ без истории"]], r_ltr3k[SL["ответ без истории"]])
        row.update(nohist_d_3k=dn3[0], nohist3_lo=dn3[1], nohist3_hi=dn3[2])
        rows.append(row)
        preds[tag] = (p, m)
        print(f"{tag:34s} n={len(ids):5d} trees={m.best_iteration_:4d} val {r.mean():.4f} "
              f"Δман {d_man[0]:+.4f} [{d_man[1]:+.4f};{d_man[2]:+.4f}] Δ3k {d_3k[0]:+.4f} [{d_3k[1]:+.4f};{d_3k[2]:+.4f}] "
              f"без ист. {row['R ответ без истории']:.4f} ({dn[0]:+.4f} [{dn[1]:+.4f};{dn[2]:+.4f}]) "
              f"es {r_es.mean():.4f}  {time.time() - t1:.0f} с", flush=True)
        pd.DataFrame(rows).to_csv(RES / "results.csv", index=False)
        return p, m

    base_rows = []
    for nm, r in (("geo2", r_geo2), ("geo2+бонусы", r_man), ("LTR-3k (2 фолда, сабмит 2)", r_ltr3k),
                  ("LTR-3k (5 фолдов)", r_ltr3k5)):
        base_rows.append({"вариант": nm, "val": r.mean(), **{"R " + k: r[mk].mean() for k, mk in SL.items()}})
    pd.DataFrame(base_rows).to_csv(RES / "bases.csv", index=False)

    import os
    stage = os.environ.get("LTRBIG_STAGE", "all")
    # 1) кривая обучения
    for tag, sub in subsets.items():
        if stage == "8k" and tag not in ("8k",):
            continue
        run(f"кривая {tag}", sub)
    if stage == "8k":
        return
    full = subsets[f"все ({len(fit_pool)})"]
    # 2) варианты против опоры на популярность
    run("все, вес истории 0.33", full, hist_w=0.33)
    run("все, только ответы без истории", full[~full["target_has_history"]])
    run("все, без признаков качества", full, feats_=[c for c in feats if c not in QUALITY])
    run("все, без весов слоёв", full, w_mode="none")
    run("все, без своих добавок", full, feats_=[c for c in feats if c not in EXTRA])
    pd.DataFrame(rows).to_csv(RES / "results.csv", index=False)
    print(f"готово за {time.time() - t0:.0f} с")


def leak_check(ft, fv, pq, vq, plabels, vlabels, RES):
    """Сравнение пула и валидации по признакам ПРАВИЛЬНЫХ ответов (среди топ-300) и по geo2 Recall.
    Если бы объекты фолда видели строки своих запросов, у пула P(mc) и loc_share правильного были бы
    заметно выше, чем у валидации (у неё объекты честно построены без её строк)."""
    out = []
    pflag = pq.set_index("query_id")
    vflag = vq.set_index("query_id")
    for nm, f, fl, lab in (("пул", ft, pflag, plabels), ("валидация", fv, vflag, vlabels)):
        pos = f[f["y"] == 1]
        r50 = recall_of(f, -f["rank"].to_numpy(float), lab)
        at300 = f.groupby("query_id")["y"].sum() / pd.Series({q: len(lab[q]) for q in f["query_id"].unique()})
        for st_name, st in (("все", None), ("текст в train", True), ("текста нет в train", False)):
            qs = fl.index if st is None else fl.index[fl["text_in_train"] == st]
            qs = qs[qs.isin(r50.index)]
            p = pos[pos["query_id"].isin(set(qs))]
            out.append({"набор": nm, "слой": st_name, "запросов": len(qs),
                        "geo2 R@50": r50.reindex(qs).mean(), "R@300": at300.reindex(qs).mean(),
                        "P(mc) правильного": p["p_mc"].mean(), "mc правильного = топ-1": (p["mc_rank"] <= np.log1p(1) + 1e-6).mean(),
                        "loc_share правильного": p["loc_share"].mean(), "same_city правильного": p["same_city"].mean(),
                        "log_reviews правильного": p["log_reviews"].mean(), "ранг geo2 правильного (медиана)": p["rank"].median(),
                        "история ответа": fl.loc[qs, "target_has_history"].mean()})
    d = pd.DataFrame(out)
    d.to_csv(RES / "leak_check.csv", index=False)
    print(d.round(4).to_string(index=False), flush=True)


def cmd_final(hist_w: float, n_models: int, tag: str):
    """Ансамбль: n_models моделей на всём пуле (разные сиды и разные 10% текстов для ранней остановки),
    среднее предсказаний. Пишет work/cands/hybrid_geo2_ltrbig_{val,bench}.parquet (топ-300)."""
    from validation import load_validation
    from ltr import to_cands
    t0 = time.time()
    RES = OUT / "res"
    RES.mkdir(exist_ok=True)
    vq, vlabels, _ = load_validation()
    pq, plabels = load_pool()
    bq = queries_of("bench")
    fv, fb = load_feats("val"), load_feats("bench")
    feats = FEATS_ALL
    ft = pd.concat([pd.read_parquet(OUT / f"feat_{s}.parquet", columns=["query_id", "rank", "y"] + feats)
                    for s in TRAIN_SETS], ignore_index=True)
    has_pos = ft.groupby("query_id", sort=False)["y"].transform("max").to_numpy() > 0
    texts = pq["text_norm"].drop_duplicates().to_numpy()
    perm = np.random.default_rng(10).permutation(len(texts))
    pv, pb, per_model = np.zeros(len(fv)), np.zeros(len(fb)), []
    r_man = cands_recall(CANDS / "hybrid_geo2_meta_val.parquet", vlabels)
    r_ltr3k = cands_recall(CANDS / "hybrid_geo2_ltr_val.parquet", vlabels)
    for k in range(n_models):
        es_texts = set(texts[perm[k::10]])          # свои 10% текстов у каждой модели
        es_ids = set(pq.loc[pq["text_norm"].isin(es_texts), "query_id"])
        is_es = ft["query_id"].isin(es_ids).to_numpy()
        sub = pq[~pq["query_id"].isin(es_ids)]
        w = pool_weights(sub, bq)
        if hist_w != 1.0:
            h = sub.set_index("query_id")["target_has_history"].reindex(w.index).to_numpy()
            w = w * np.where(h, hist_w, 1.0)
            w = w / w.mean()
        m = fit_ranker(ft[~is_es & has_pos], ft[is_es & has_pos], feats, w, seed=k)
        a = m.predict(fv[feats].to_numpy(np.float32))
        b = m.predict(fb[feats].to_numpy(np.float32))
        # скоры lambdarank у разных моделей в разной шкале: усредняем z-оценки внутри запроса
        za = _zq(fv, a)
        pv += za
        pb += _zq(fb, b)
        r = recall_of(fv, a, vlabels)
        per_model.append(r.mean())
        print(f"модель {k}: деревьев {m.best_iteration_}, val {r.mean():.4f}; {time.time() - t0:.0f} с", flush=True)
    pv /= n_models
    pb /= n_models
    r = recall_of(fv, pv, vlabels)
    d1, d2 = boot(r, r_man), boot(r, r_ltr3k)
    nh = ~vq.set_index("query_id").loc[r.index, "target_has_history"].to_numpy()
    d3, d4 = boot(r[nh], r_man[nh]), boot(r[nh], r_ltr3k[nh])
    print(f"ансамбль {n_models}: val {r.mean():.4f} (модели {np.round(per_model, 4)}); "
          f"Δ к бонусам {d1[0]:+.4f} [{d1[1]:+.4f};{d1[2]:+.4f}], Δ к LTR-3k {d2[0]:+.4f} [{d2[1]:+.4f};{d2[2]:+.4f}]; "
          f"без истории {r[nh].mean():.4f}: Δ к бонусам {d3[0]:+.4f} [{d3[1]:+.4f};{d3[2]:+.4f}], "
          f"Δ к LTR-3k {d4[0]:+.4f} [{d4[1]:+.4f};{d4[2]:+.4f}]", flush=True)
    r.rename("recall@50").to_frame().to_parquet(RES / f"val_per_query_{tag}.parquet")
    cv_, cb = to_cands(fv, pv), to_cands(fb, pb)
    assert cv_.groupby("query_id").size().eq(300).all() and cv_["query_id"].nunique() == 3000
    assert cb.groupby("query_id").size().eq(300).all() and cb["query_id"].nunique() == len(bq)
    suffix = "" if tag == "main" else f"_{tag}"
    cv_.to_parquet(CANDS / f"hybrid_geo2_ltrbig{suffix}_val.parquet", index=False)
    cb.to_parquet(CANDS / f"hybrid_geo2_ltrbig{suffix}_bench.parquet", index=False)
    chk = cands_recall(CANDS / f"hybrid_geo2_ltrbig{suffix}_val.parquet", vlabels)
    print(f"файл val: Recall@50 = {chk.mean():.4f}")
    # совпадение топ-50 на бенчмарке с прежними файлами и сдвиг признаков топ-50 бенчмарк / валидация
    top = cb[cb["rank"] <= 50]
    for nm in ("hybrid_geo2_bench", "hybrid_geo2_meta_bench", "hybrid_geo2_ltr_bench"):
        o = pd.read_parquet(CANDS / f"{nm}.parquet")
        print(f"  бенчмарк: топ-50 совпадает с {nm} на {top.merge(o[o['rank'] <= 50], on=['query_id', 'item_id']).shape[0] / len(top):.3f}")
    sh = []
    for nm, f, c in (("val", fv, cv_), ("bench", fb, cb)):
        t50 = c[c["rank"] <= 50][["query_id", "item_id"]].merge(f[["query_id", "item_id"] + feats], on=["query_id", "item_id"])
        sh.append(t50[["log_reviews", "desc_len", "p_mc", "same_city", "loc_share", "g_rank", "phrase_text"]].mean().rename(nm))
    print(pd.concat(sh, axis=1).round(3).to_string())
    print(f"готово за {time.time() - t0:.0f} с")


def cmd_leakdiag():
    """Диагностика утечки.
    1) Сопоставимые слои: тексты пула, которые сами по себе единственные (режим unseen_nat, 1 запрос на текст),
       против «текста нет в train» валидации; тексты пула seen против «текст в train» валидации (по 1 запросу
       на текст). 2) Калибровка: P(mc) правильного ответа от классификатора, ВИДЕВШЕГО строки своих запросов
       (обучен на train[mask] целиком для пула фолда 0; на полном train для валидации), против честного.
       Если бы схема фолдов протекала, честные значения пула были бы близки к «протёкшим»."""
    import rerank_meta as rm
    from validation import load_validation
    RES = OUT / "res"
    RES.mkdir(exist_ok=True)
    vq, vlabels, mask = load_validation()
    pq, plabels = load_pool()
    cols = ["query_id", "item_id", "rank", "y", "p_mc", "loc_share", "same_city", "log_reviews"]
    ft = pd.concat([pd.read_parquet(OUT / f"feat_{s}.parquet", columns=cols) for s in TRAIN_SETS], ignore_index=True)
    fv = pd.read_parquet(OUT / "feat_val.parquet", columns=cols)
    one = pq.sample(frac=1.0, random_state=3).drop_duplicates("text_norm")
    groups_ = {
        "пул: unseen_nat, 1 запрос на текст": one[(one["mode"] == "unseen_nat") & (one["n_text"] == 1)],
        "пул: unseen_forced, 1 на текст": one[one["mode"] == "unseen_forced"],
        "пул: seen, 1 на текст": one[one["mode"] == "seen"],
        "валидация: текста нет в train": vq[~vq["text_in_train"]],
        "валидация: текст в train": vq[vq["text_in_train"]],
    }
    rows = []
    for nm, g in groups_.items():
        f = fv if nm.startswith("валидация") else ft
        lab = vlabels if nm.startswith("валидация") else plabels
        ids = set(g["query_id"])
        ff = f[f["query_id"].isin(ids)]
        r50 = recall_of(ff, -ff["rank"].to_numpy(float), lab)
        pos = ff[ff["y"] == 1]
        rows.append({"группа": nm, "запросов": len(ids), "geo2 R@50": r50.mean(),
                     "P(mc) правильного": pos["p_mc"].mean(), "loc_share правильного": pos["loc_share"].mean(),
                     "история": g["target_has_history"].mean(), "слов": g["text_norm"].str.split().str.len().mean(),
                     "с фильтром": g["has_filter"].mean(), "регион": g["is_region"].mean()})
    d = pd.DataFrame(rows)
    print(d.round(4).to_string(index=False), flush=True)
    d.to_csv(RES / "leak_matched.csv", index=False)

    # калибровка «протёкшим» классификатором
    tr = load_train_full()[["search_query", "search_infm_params_text", "item_microcat_id"]]
    items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_microcat_id"])
    items["item_id"] = items["item_id"].astype(str)
    mc_of = dict(zip(items["item_id"], items["item_microcat_id"]))
    out = []
    for nm, q, lab, f, fit in (("пул, фолд 0", pq[pq["fold"] == 0], plabels, ft, tr[mask]),
                               ("валидация", vq, vlabels, fv, tr)):
        m = rm.fit_microcat(fit)
        proba = m.predict_proba(q)
        ci = {c: i for i, c in enumerate(m.classes_)}
        qi = {x: k for k, x in enumerate(q["query_id"])}
        # те же пары, что в честных признаках: правильные ответы в топ-300
        pos = f[(f["y"] == 1) & f["query_id"].isin(set(q["query_id"]))].copy()
        j = pos["item_id"].map(mc_of).map(ci)
        ok = j.notna().to_numpy()
        lk = np.zeros(len(pos))
        lk[ok] = proba[pos["query_id"].map(qi).to_numpy()[ok], j[ok].astype(int).to_numpy()]
        pos["leaky"] = lk
        fl = q.set_index("query_id")
        key = "mode" if "mode" in q else "text_in_train"
        pos["слой"] = pos["query_id"].map(fl[key]).astype(str) + " / фильтр=" + pos["query_id"].map(fl["has_filter"]).astype(str)
        for sl, g in [("все", pos)] + list(pos.groupby("слой")):
            out.append({"набор": nm, "слой": sl, "пар": len(g), "P(mc) честно": g["p_mc"].mean(),
                        "P(mc), классификатор видел свои строки": g["leaky"].mean()})
            print(out[-1], flush=True)
        del m
        gc.collect()
    pd.DataFrame(out).to_csv(RES / "leak_calib.csv", index=False)


# ====================================================================== дообученный эмбеддер

def ft_query_emb(name: str, split: str) -> np.ndarray:
    """Эмбеддинги запросов дообученной моделью (кэш work/ltr_big/qft_<name>_<split>.npy).
    Веса читаются сразу в fp16 на CPU и переносятся на GPU (пик VRAM ~1.1 ГБ у bge-m3), модель выгружается."""
    path = OUT / f"qft_{name}_{split}.npy"
    if path.exists():
        return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer
    from ft_eval import MODELS_DIR, encode_q, prefixes
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        torch.cuda.set_per_process_memory_fraction(min(1.0, 2.0 / total))
    m = SentenceTransformer(str(MODELS_DIR / name), device="cpu", model_kwargs={"dtype": torch.float16}).to("cuda")
    m.max_seq_length = 256
    e = encode_q(m, queries_of(split), prefixes(name)[0])
    del m
    gc.collect()
    torch.cuda.empty_cache()
    np.save(path, e)
    return e


def cmd_ftcands(name: str):
    """Кандидаты «BM25 + bge-m3 + дообученный dense» (взвешенный RRF 1 / 0.5 / 0.5, та же гео-схема) для
    валидации и бенчмарка — как вариант bge+<name>_w в src/ft_eval.py. Только val/bench: дообученная модель
    учились на train[mask] и видела строки запросов пула, поэтому для обучения LTR её сигнал не используется.
    Эмбеддинги bge-m3 запросов — из шага retrieve (work/ltr_big/qemb_*.npy), модель bge-m3 не грузится."""
    import torch
    from dense import load_corpus_emb
    from ft_eval import encode_q, load_ft, prefixes, rule_for, run_multi
    from geo import load_geo_data
    from geo_run import build_geo, save_cands
    from pipeline import Config, build_sparse_index
    from validation import load_validation, region_location_ids
    qft = {sp_: ft_query_emb(name, sp_) for sp_ in ("val", "bench")}     # до загрузки корпусов на GPU
    t0 = time.time()
    items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_title_raw"])
    items["item_id"] = items["item_id"].astype(str)
    sp = build_sparse_index(items, Config())
    ids = items["item_id"].to_numpy()
    del items
    eb, ib = load_corpus_emb("bge-m3_s256")
    ef, i_f = load_corpus_emb(f"{name}_s256")
    assert (ib.astype(str) == ids).all() and (i_f.astype(str) == ids).all()
    Db = torch.from_numpy(eb).to("cuda", torch.float16)
    Df = torch.from_numpy(ef).to("cuda", torch.float16)
    del eb, ef
    corpus, train = load_geo_data()
    mask = load_validation()[2]
    regions = region_location_ids()
    for split in ("val", "bench"):
        q = queries_of(split)
        qf = qft[split]
        qb = np.load(OUT / f"qemb_{split}.npy")
        maps = build_geo(corpus, train, train if split == "bench" else train[mask], regions)
        kept = run_multi(q, rule_for(q["search_location_id"].to_numpy(), regions), sp, [(Db, qb), (Df, qf)],
                         maps, [1.0, 0.5, 0.5])
        c = save_cands(kept, q, ids, OUT / f"fthyb_{name}_{split}.parquet")
        print(f"{split}: {c['query_id'].nunique()} запросов за {time.time() - t0:.0f} с", flush=True)
        ref = WORK / "ft" / f"raw_bge+{name}_w_{split}.parquet"
        if ref.exists():
            r = pd.read_parquet(ref)
            a = c[c["rank"] <= 50].groupby("query_id")["item_id"].agg(frozenset)
            b = r[r["rank"] <= 50].groupby("query_id")["item_id"].agg(frozenset)
            print(f"  сверка с {ref.name}: топ-50 совпадает у {(a == b.reindex(a.index)).mean():.4f} запросов")


def fuse(ltr: pd.DataFrame, ft: pd.DataFrame, w: float, k: int = 60) -> pd.DataFrame:
    """Позднее слияние: score = 1/(k + ранг LTR) + w/(k + ранг в списке с дообученным dense).
    Кандидаты — объединение двух топ-300; ответ — первые 300 по новому скору."""
    a = ltr[["query_id", "item_id", "rank"]].rename(columns={"rank": "r1"})
    b = ft[["query_id", "item_id", "rank"]].rename(columns={"rank": "r2"})
    m = a.merge(b, on=["query_id", "item_id"], how="outer")
    sc = np.nan_to_num(1.0 / (k + m["r1"].to_numpy(float)), nan=0.0) + \
        w * np.nan_to_num(1.0 / (k + m["r2"].to_numpy(float)), nan=0.0)
    m["score"] = sc.astype(np.float32)
    m["old"] = m["r1"].fillna(10_000) + m["r2"].fillna(10_000) / 1e5
    m = m.sort_values(["query_id", "score", "old"], ascending=[True, False, True], kind="mergesort")
    m["rank"] = (m.groupby("query_id", sort=False).cumcount() + 1).astype(np.int32)
    m = m[m["rank"] <= 300]
    return m[["query_id", "item_id", "score", "rank"]].reset_index(drop=True)


def cmd_fuse(name: str):
    """Вес w выбирается на половине A валидации (seed 0, как в 11/16), отчёт — на B и на всех 3000."""
    from validation import load_validation
    RES = OUT / "res"
    vq, vlabels, _ = load_validation()
    ltr_v = pd.read_parquet(CANDS / "hybrid_geo2_ltrbig_val.parquet")
    ltr_b = pd.read_parquet(CANDS / "hybrid_geo2_ltrbig_bench.parquet")
    ft_v = pd.read_parquet(OUT / f"fthyb_{name}_val.parquet")
    ft_b = pd.read_parquet(OUT / f"fthyb_{name}_bench.parquet")
    qids = np.array(sorted(vlabels))
    perm = np.random.default_rng(0).permutation(len(qids))
    isB = pd.Series(False, index=qids)
    isB.iloc[perm[len(qids) // 2:]] = True
    r_man = cands_recall(CANDS / "hybrid_geo2_meta_val.parquet", vlabels)
    r_ltr = cands_recall(CANDS / "hybrid_geo2_ltrbig_val.parquet", vlabels)
    r_ft = cands_recall(WORK / "ft" / f"meta_bge+{name}_w_val.parquet", vlabels) \
        if (WORK / "ft" / f"meta_bge+{name}_w_val.parquet").exists() else None
    nh = ~vq.set_index("query_id").loc[qids, "target_has_history"].to_numpy()
    rows, per = [], {}
    for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
        c = fuse(ltr_v, ft_v, w)
        top = c[c["rank"] <= 50].groupby("query_id")["item_id"].agg(set)
        r = pd.Series({q: len(top.get(q, set()) & v) / len(v) for q, v in vlabels.items()}).sort_index()
        per[w] = r
        rows.append(dict(w=w, A=r[~isB.to_numpy()].mean(), B=r[isB.to_numpy()].mean(), все=r.mean(),
                         без_истории=r[nh].mean()))
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    w_best = float(g.loc[g["A"].idxmax(), "w"])
    r = per[w_best]
    B = isB.to_numpy()
    out = {"w (выбран на A)": w_best}
    for nm, base in (("к LTR-big", r_ltr), ("к geo2+бонусы", r_man)) + ((("к bge+ft_w+бонусы", r_ft),) if r_ft is not None else ()):
        d_all, d_B, d_nh = boot(r, base), boot(r[B], base[B]), boot(r[nh], base[nh])
        out[nm] = (f"все {d_all[0]:+.4f} [{d_all[1]:+.4f};{d_all[2]:+.4f}], B {d_B[0]:+.4f} [{d_B[1]:+.4f};{d_B[2]:+.4f}], "
                   f"без истории {d_nh[0]:+.4f} [{d_nh[1]:+.4f};{d_nh[2]:+.4f}]")
    print(g.round(4).to_string(index=False))
    print(f"итог: R@50 все {r.mean():.4f}, B {r[B].mean():.4f}, без истории {r[nh].mean():.4f}; LTR-big {r_ltr.mean():.4f}"
          + (f"; bge+ft_w+бонусы {r_ft.mean():.4f}" if r_ft is not None else ""))
    for k_, v_ in out.items():
        print(f"  {k_}: {v_}")
    g.to_csv(RES / f"fuse_{name}.csv", index=False)
    # срезы
    d = vq.set_index("query_id").loc[qids]
    SL = {"все": np.ones(len(d), bool), "текста нет в train": ~d.text_in_train.to_numpy(),
          "регион или вся Россия": d.is_region.to_numpy(), "ответ без истории": nh,
          "у ответа есть история": ~nh, "с фильтром": d.has_filter.to_numpy(), "без фильтра": ~d.has_filter.to_numpy()}
    sl = pd.DataFrame({k_: {"n": int(mk.sum()), "LTR-big": r_ltr[mk].mean(), "LTR-big+ft": r[mk].mean(),
                            **({"bge+ft_w+бонусы": r_ft[mk].mean()} if r_ft is not None else {})}
                       for k_, mk in SL.items()}).T
    print(sl.round(4).to_string())
    sl.to_csv(RES / f"fuse_slices_{name}.csv")
    cv_, cb = fuse(ltr_v, ft_v, w_best), fuse(ltr_b, ft_b, w_best)
    assert cv_.groupby("query_id").size().eq(300).all() and cv_["query_id"].nunique() == 3000
    assert cb.groupby("query_id").size().eq(300).all() and cb["query_id"].nunique() == ltr_b["query_id"].nunique()
    # итоговые файлы с ft — от стекера (cmd_stack); слияние пишется только в work/ltr_big/
    cv_.to_parquet(OUT / f"fuse_{name}_val.parquet", index=False)
    cb.to_parquet(OUT / f"fuse_{name}_bench.parquet", index=False)
    chk = cands_recall(OUT / f"fuse_{name}_val.parquet", vlabels)
    top = cb[cb["rank"] <= 50]
    o = ltr_b[ltr_b["rank"] <= 50]
    print(f"файл val R@50 = {chk.mean():.4f}; бенчмарк: топ-50 совпадает с LTR-big на "
          f"{top.merge(o, on=['query_id', 'item_id']).shape[0] / len(top):.3f}")


def _stack_frame(split: str, names: list[str], q_ft: dict, Df: dict, pos_of: dict) -> pd.DataFrame:
    """Кандидаты стекера: объединение топ-300 LTR-big и топ-300 списков «bge + дообученный dense» (по моделям).
    Признаки: z-оценка и ранг LTR-big (NaN, если кандидата нет в его топ-300); для каждой дообученной модели —
    ранг и скор в её списке, косинус (сырой, разница с максимумом в запросе, ранг в объединении)."""
    q = queries_of(split)
    m = pd.read_parquet(CANDS / f"hybrid_geo2_ltrbig_{split}.parquet").rename(columns={"rank": "ltr_rank", "score": "ltr_score"})
    for j, nm in enumerate(names):
        sfx = "" if j == 0 else f"_{j}"
        b = pd.read_parquet(OUT / f"fthyb_{nm}_{split}.parquet").rename(columns={"rank": "ft_rank" + sfx, "score": "ft_score" + sfx})
        m = m.merge(b, on=["query_id", "item_id"], how="outer")
    qi = {x: k for k, x in enumerate(q["query_id"])}
    order = m["query_id"].map(qi).to_numpy()
    m = m.iloc[np.lexsort((m["ltr_rank"].fillna(1e4).to_numpy(), order))].reset_index(drop=True)
    qrow = m["query_id"].map(qi).to_numpy()
    irow = m["item_id"].map(pos_of).to_numpy()
    g = m.groupby("query_id", sort=False)
    for j, nm in enumerate(names):
        sfx = "" if j == 0 else f"_{j}"
        m["ft_cos" + sfx] = np.einsum("ij,ij->i", q_ft[nm][qrow], Df[nm][irow].astype(np.float32)).astype(np.float32)
        m["ft_cos_gap" + sfx] = (m["ft_cos" + sfx] - g["ft_cos" + sfx].transform("max")).astype(np.float32)
        m["ft_cos_rank" + sfx] = g["ft_cos" + sfx].rank(ascending=False, method="first").astype(np.float32)
    m["ltr_z"] = ((m["ltr_score"] - g["ltr_score"].transform("mean")) / (g["ltr_score"].transform("std") + 1e-9)).astype(np.float32)
    m["in_ltr"] = m["ltr_rank"].notna().astype(np.int8)
    m["rank"] = m["ltr_rank"].fillna(1000).astype(np.int32)     # для recall50: тай-брейк по LTR
    return m


STACK_FT = ["ft_rank", "ft_score", "ft_cos", "ft_cos_gap", "ft_cos_rank"]
STACK_BASE = ["ltr_z", "ltr_rank", "in_ltr"]


def cmd_stack(name: str):
    """Вариант (в): стекер поверх LTR-big, обучаемый ТОЛЬКО на 3000 запросах валидации перекрёстно
    (дообученный эмбеддер их не видел). Контроль — тот же стекер без признаков дообученной модели.
    Бенчмарк: стекер на всех 3000."""
    names = name.split(",")
    import torch
    import lightgbm as lgb
    from dense import load_corpus_emb
    from ft_eval import encode_q, load_ft, prefixes
    from ltr import to_cands
    from validation import load_validation
    RES = OUT / "res"
    vq, vlabels, _ = load_validation()
    Df = {}
    for nm in names:
        Df[nm], ids = load_corpus_emb(f"{nm}_s256")
    pos_of = {x: i for i, x in enumerate(ids.astype(str))}
    fr = {}
    for split in ("val", "bench"):
        fr[split] = _stack_frame(split, names, {nm: ft_query_emb(nm, split) for nm in names}, Df, pos_of)
    del Df
    stack_ft = [c + ("" if j == 0 else f"_{j}") for j in range(len(names)) for c in STACK_FT]
    fv, fb = fr["val"], fr["bench"]
    fv["y"] = np.fromiter((i in vlabels[q] for q, i in zip(fv["query_id"], fv["item_id"])), bool, len(fv)).astype(np.int8)
    at = fv.groupby("query_id")["y"].sum() / pd.Series({q: len(v) for q, v in vlabels.items()})
    print(f"объединение кандидатов: {len(fv) / 3000:.0f} на запрос, R@всех = {at.mean():.4f}")
    qids = np.array(sorted(vlabels))
    perm = np.random.default_rng(0).permutation(len(qids))
    fold5 = pd.Series(np.arange(len(qids))[np.argsort(perm)] % 5, index=qids)
    fold2 = pd.Series((np.arange(len(qids))[np.argsort(perm)] >= len(qids) // 2).astype(int), index=qids)
    params = dict(objective="lambdarank", lambdarank_truncation_level=60, n_estimators=300, learning_rate=0.03,
                  num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=1.0,
                  reg_lambda=1.0, random_state=0, n_jobs=8, verbose=-1)
    r_ltr = cands_recall(CANDS / "hybrid_geo2_ltrbig_val.parquet", vlabels)
    nh = ~vq.set_index("query_id").loc[qids, "target_has_history"].to_numpy()
    res, oofs = [], {}
    for tag, feats in (("стекер без ft (контроль)", STACK_BASE), ("стекер с ft", STACK_BASE + stack_ft)):
        for fname, folds in (("2 фолда", fold2), ("5 фолдов", fold5)):
            fo = fv["query_id"].map(folds).to_numpy()
            oof = np.zeros(len(fv))
            for k in np.unique(fo):
                tr, te = fo != k, fo == k
                ftr = fv[tr]
                m = lgb.LGBMRanker(**params)
                m.fit(ftr[feats].to_numpy(np.float32), ftr["y"].to_numpy(), group=groups(ftr["query_id"].to_numpy()))
                oof[te] = m.predict(fv.loc[te, feats].to_numpy(np.float32))
            r = recall_of(fv, oof, vlabels)
            oofs[(tag, fname)] = (r, oof)
            d = boot(r, r_ltr)
            dn = boot(r[nh], r_ltr[nh])
            res.append(dict(вариант=tag, схема=fname, R50=r.mean(), без_истории=r[nh].mean(), d_ltr=d[0], lo=d[1], hi=d[2],
                            d_nh=dn[0], nh_lo=dn[1], nh_hi=dn[2]))
            print(f"{tag:26s} {fname}: R@50 {r.mean():.4f} (без истории {r[nh].mean():.4f}); Δ к LTR-big {d[0]:+.4f} "
                  f"[{d[1]:+.4f};{d[2]:+.4f}], без истории {dn[0]:+.4f} [{dn[1]:+.4f};{dn[2]:+.4f}]", flush=True)
    # ft против контроля на тех же фолдах (чистый вклад дообученной модели)
    for fname in ("2 фолда", "5 фолдов"):
        a_, b_ = oofs[("стекер с ft", fname)][0], oofs[("стекер без ft (контроль)", fname)][0]
        d, dn = boot(a_, b_), boot(a_[nh], b_[nh])
        print(f"вклад ft ({fname}): {d[0]:+.4f} [{d[1]:+.4f};{d[2]:+.4f}], без истории {dn[0]:+.4f} [{dn[1]:+.4f};{dn[2]:+.4f}]")
        res.append(dict(вариант="вклад ft (с ft − контроль)", схема=fname, d_ltr=d[0], lo=d[1], hi=d[2], d_nh=dn[0], nh_lo=dn[1], nh_hi=dn[2]))
    pd.DataFrame(res).to_csv(RES / f"stack_{name.replace(',', '+')}.csv", index=False)
    # файлы: val — вне-фолдовые скоры (5 фолдов) стекера с ft; bench — стекер на всех 3000
    feats = STACK_BASE + stack_ft
    oof = oofs[("стекер с ft", "5 фолдов")][1]
    m = lgb.LGBMRanker(**params)
    m.fit(fv[feats].to_numpy(np.float32), fv["y"].to_numpy(), group=groups(fv["query_id"].to_numpy()))
    pb = m.predict(fb[feats].to_numpy(np.float32))
    cv_ = to_cands(fv, oof)
    cb = to_cands(fb, pb)
    cv_ = cv_[cv_["rank"] <= 300].reset_index(drop=True)
    cb = cb[cb["rank"] <= 300].reset_index(drop=True)
    assert cv_.groupby("query_id").size().eq(300).all() and cb.groupby("query_id").size().eq(300).all()
    tag = "" if name == "e5-small-ft" else "_" + name.replace(",", "+")
    cv_.to_parquet(OUT / f"stack{tag}_val.parquet", index=False)
    cb.to_parquet(OUT / f"stack{tag}_bench.parquet", index=False)
    print(f"файлы стекера: {OUT}/stack{tag}_{{val,bench}}.parquet; val R@50 = {cands_recall(OUT / f'stack{tag}_val.parquet', vlabels).mean():.4f}")


def _zq(f: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """z-оценка предсказания внутри запроса."""
    s = pd.Series(p)
    g = s.groupby(f["query_id"].to_numpy())
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy()


def main():
    """Разбор аргументов командной строки и запуск шага."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "retrieve", "features", "train", "final", "leakdiag", "ftcands", "fuse", "stack"])
    ap.add_argument("--hist-w", type=float, default=1.0, help="final: вес запросов, у ответа которых есть история")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ft-name", default="e5-small-ft", help="ftcands/fuse: имя дообученной модели в work/models")
    ap.add_argument("--splits", default="val,bench," + ",".join(TRAIN_SETS))
    a = ap.parse_args()
    splits = a.splits.split(",")
    if a.cmd == "sample":
        cmd_sample()
    elif a.cmd == "retrieve":
        cmd_retrieve(splits)
    elif a.cmd == "features":
        cmd_features(splits)
    elif a.cmd == "train":
        cmd_train()
    elif a.cmd == "leakdiag":
        cmd_leakdiag()
    elif a.cmd == "ftcands":
        cmd_ftcands(a.ft_name)
    elif a.cmd == "fuse":
        cmd_fuse(a.ft_name)
    elif a.cmd == "stack":
        cmd_stack(a.ft_name)
    else:
        cmd_final(a.hist_w, a.n_models, a.tag)


if __name__ == "__main__":
    main()
