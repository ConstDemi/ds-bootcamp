"""LTR-big + дообученный эмбеддер bge-m3-lora, честно для обучения LTR (перекрёстное дообучение).

Проблема (eda/18_ltr_big.md): bge-m3-lora училась на всём train[mask] и видела запросы пула LTR,
поэтому её косинус нельзя давать LTR как признак — на пуле он «знает ответ», на бенчмарке нет.
Решение: 5 копий модели, копия f учится без строк фолда f (ft_train.py --excl-fold f, на RunPod),
и для запроса пула из фолда f всё «дообученное» считается моделью f. Для валидации и бенчмарка —
bge-m3-lora на всём train[mask] (валидационных запросов в нём нет, как и у копий — строк своего фолда).

Шаги (запуск из корня репозитория; ltr_big.py и ltr.py не меняются, только импортируются):
    python src/ltr_ft.py retrieve [--splits ...]   # кандидаты geo2 (как ltr_big) ∪ топ-K дообученной модели в том же
                                                   # гео-пуле + скоры источников -> work/ltr_ft/cands_*.parquet
    python src/ltr_ft.py features [--splits ...]   # признаки ltr_big + признаки дообученной модели -> feat_*.parquet
    python src/ltr_ft.py leak                      # проверка утечки: косинус дообученной модели у правильного ответа,
                                                   # пул (модель своего фолда) против валидации, сопоставимые слои
    python src/ltr_ft.py final [--variants ...]    # ансамбль 5 LGBMRanker как ltr_big final, сравнение с 0.901,
                                                   # -> work/cands/hybrid_ltrft_{val,bench}.parquet

Входы: work/emb/bge-m3-lora-f{f}_s256.npy (корпус моделью f), work/ltr_big/qft_bge-m3-lora-f{f}_tr{f}.npy
(запросы пула фолда f моделью f) — считаются на GPU (ft_eval.py encode, ltr_big.ft_query_emb);
work/emb/bge-m3-lora_s256.npy и work/ltr_big/qft_bge-m3-lora_{val,bench}.npy — для валидации и бенчмарка.
Проверка кода без моделей фолдов: --fold-model bge-m3-lora --limit 300 --out work/ltr_ft_smoke
(все фолды берут общую модель — это утечка, цифры такого прогона ничего не значат).
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

import ltr_big as lb  # noqa: E402
from ltr_big import CANDS, DATA, OUT as LB_OUT, TRAIN_SETS, WORK  # noqa: E402

FT_VAL = "bge-m3-lora"            # модель для валидации и бенчмарка (train[mask])
FT_FOLD = "bge-m3-lora-f{f}"      # модель для запросов пула фолда f (train[mask] без фолда f)
K_FT = 100                        # сколько кандидатов добирает дообученная модель внутри гео-пула
N_FUSED = 2000                    # сколько позиций RRF-списка geo2 хранить со «своим» рангом (для новых кандидатов)

# признаки дообученной модели (добавляются к FEATURES + EXTRA из ltr_big)
FT_FEATS = ["ft_cos", "ft_cos_grank", "ft_rank", "ft_cos_pct", "ft_cos_gap", "ft_cos_z", "is_new"]

MODELS_OUT = WORK / "ltr_ft" / "models"   # 5 бустеров итоговой схемы + ltr.json (признаки, параметры)

ARGS = argparse.Namespace(out=WORK / "ltr_ft", fold_model=FT_FOLD, val_model=FT_VAL, limit=0)


def out_dir() -> Path:
    """Папка результатов (по умолчанию work/ltr_ft), создаётся при необходимости."""
    p = Path(ARGS.out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_for(split: str) -> str:
    """Какая дообученная модель считает косинус для запросов набора split."""
    if split in ("val", "bench"):
        return ARGS.val_model
    return ARGS.fold_model.format(f=int(split[2:]))


def queries(split: str) -> pd.DataFrame:
    """Запросы набора (с флагами), первые --limit при проверке кода."""
    q = lb.queries_of(split)
    return q.head(ARGS.limit) if ARGS.limit else q


# ====================================================================== шаг 1: кандидаты

CAND_COLS = (("score", np.float32), ("rank", np.int32), ("bm25", np.float32), ("bm25_grank", np.int32),
             ("cos", np.float32), ("cos_grank", np.int32), ("ft_cos", np.float32), ("ft_cos_grank", np.int32),
             ("ft_rank", np.int32), ("is_new", np.int8))


def union_candidates(q: pd.DataFrame, q_emb: np.ndarray, q_ft: np.ndarray, sparse, D, Dft, maps,
                     item_ids: np.ndarray, cfg, tag: str = "") -> pd.DataFrame:
    """Кандидаты = топ-300 geo2 (побитово как ltr_big retrieve / pipeline.retrieve_candidates) ∪ топ-K_FT по
    косинусу дообученной модели в том же гео-пуле (ярусы и добор из корпуса — как у dense в geo2).

    q_emb / q_ft — эмбеддинги запросов готовой bge-m3 и дообученной модели (строки = строки q);
    D / Dft — эмбеддинги корпуса ими же (torch, на GPU в fp16); maps — GeoMaps по train_fit набора.
    Для новых кандидатов g_rank/g_score честные, если объявление есть в продолжении RRF-списка geo2
    (до N_FUSED позиций: RRF-скор + гео-бонус); если его нет ни в одном списке источников — скор равен
    гео-бонусу, ранг после всего списка по порядку косинуса. Колонки источников (bm25, cos и их глобальные
    ранги) — как в ltr_big; плюс ft_cos, ft_cos_grank (глобальный ранг в корпусе), ft_rank (место в топ-K_FT
    дообученной модели внутри пула, K_FT+1 если нет), is_new (кандидата не было в топ-300 geo2)."""
    import torch
    from geo import rrf, tiered_top
    from pipeline import geo_rule
    from sparse import SparseIndex
    N, K, bs = len(item_ids), cfg.n_cands, cfg.dense_batch
    all_pos = np.arange(N)
    dev, dt = D.device, D.dtype
    locs, texts = q["search_location_id"].to_numpy(), q["search_query"].to_numpy()
    rows = {k: [] for k in ["pos"] + [c for c, _ in CAND_COLS]}
    n_per, t1 = [], time.time()
    for b0 in range(0, len(q), bs):
        with torch.no_grad():
            dsc = (torch.from_numpy(q_emb[b0:b0 + bs]).to(dev, dt) @ D.T).float().cpu().numpy()
            fsc = (torch.from_numpy(q_ft[b0:b0 + bs]).to(dev, dt) @ Dft.T).float().cpu().numpy()
        for j, (dv, fv) in enumerate(zip(dsc, fsc)):
            i = b0 + j
            loc = int(locs[i])
            ssc = sparse.scores(texts[i])
            # --- geo2 ровно как ltr_big.cmd_retrieve / pipeline.retrieve_candidates
            g_s = SparseIndex._top(all_pos, ssc, 2 * K)
            g_d = np.argpartition(-dv, 2 * K)[:2 * K]
            g_d = g_d[np.argsort(-dv[g_d], kind="stable")]
            rule = geo_rule(loc, maps.regions, cfg)
            pool = maps.tiers(maps.base_rule(rule), loc)
            top_sparse, _ = tiered_top(ssc, pool, K, g_s, positive=True)
            top_dense, _ = tiered_top(dv, pool, K, g_d, positive=False)
            bonus_mask, bonus_w = maps.bonus_mask(rule, loc)
            fused, fsc_ = rrf([top_sparse, top_dense], k_rrf=cfg.rrf_k, n_out=N_FUSED,
                              bonus=bonus_mask, bonus_w=bonus_w)
            # --- топ-K_FT дообученной модели в том же гео-пуле
            g_f = np.argpartition(-fv, 2 * K_FT)[:2 * K_FT]
            g_f = g_f[np.argsort(-fv[g_f], kind="stable")]
            top_ft, _ = tiered_top(fv, pool, K_FT, g_f, positive=False)
            # --- объединение: топ-300 geo2, затем новые из списка дообученной модели
            head = fused[:K]
            new = top_ft[~np.isin(top_ft, head)]
            pos = np.concatenate([head, new])
            where = {p: r for r, p in enumerate(fused)}          # место в продолжении RRF-списка
            n_fused = len(fused)
            rank, score = np.empty(len(pos), np.int32), np.empty(len(pos), np.float32)
            rank[:K], score[:K] = np.arange(1, K + 1), fsc_[:K]
            bw = bonus_w * bonus_mask[new] if (bonus_mask is not None and bonus_w) else np.zeros(len(new))
            for t, p in enumerate(new):
                r = where.get(p)
                if r is not None:
                    rank[K + t], score[K + t] = r + 1, fsc_[r]
                else:
                    rank[K + t], score[K + t] = n_fused + 1 + t, bw[t]
            ft_r = {p: r + 1 for r, p in enumerate(top_ft)}
            rows["pos"].append(pos)
            rows["score"].append(score)
            rows["rank"].append(rank)
            rows["is_new"].append(np.r_[np.zeros(K, np.int8), np.ones(len(new), np.int8)])
            rows["ft_rank"].append(np.array([ft_r.get(p, K_FT + 1) for p in pos], np.int32))
            # --- скоры источников: значение и глобальный ранг (1 + число объявлений со скором выше)
            for col, sc in (("cos", dv), ("ft_cos", fv)):
                v = sc[pos]
                rows[col].append(v)
                rows[col + "_grank"].append(N - np.searchsorted(np.sort(sc), v, side="right") + 1)
            b = ssc[pos]
            ps = np.sort(ssc[ssc > 0])
            rows["bm25"].append(b)
            rows["bm25_grank"].append(len(ps) - np.searchsorted(ps, b, side="right") + 1)
            n_per.append(len(pos))
        if (b0 // bs) % 20 == 0:
            print(f"  {tag}: {min(b0 + bs, len(q))}/{len(q)} за {time.time() - t1:.0f} с", flush=True)
    c = pd.DataFrame({"query_id": np.repeat(q["query_id"].to_numpy(), n_per),
                      "item_id": item_ids[np.concatenate(rows["pos"])].astype(str)})
    for col, dtp in CAND_COLS:
        c[col] = np.concatenate(rows[col]).astype(dtp)
    return c


def cmd_retrieve(splits: list[str]):
    """union_candidates для наборов: эмбеддинги запросов bge-m3 — из ltr_big retrieve (work/ltr_big/qemb_*.npy),
    дообученной моделью — кэш work/ltr_big/qft_*.npy; у запросов пула фолда f — модель фолда f."""
    import torch
    from dense import load_corpus_emb
    from pipeline import Config, build_geo_maps, build_sparse_index, load_corpus_embeddings
    from validation import load_validation, region_location_ids
    cfg = Config()
    OUT = out_dir()
    t0 = time.time()
    # эмбеддинги запросов дообученной моделью — до загрузки корпусов на GPU (кэш work/ltr_big/qft_*.npy)
    qft = {s: lb.ft_query_emb(model_for(s), s) for s in splits}
    items = pd.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id", "item_location_id", "item_latitude", "item_longitude",
                                     "item_title_raw"])
    items["item_id"] = items["item_id"].astype(str)
    items["lat"] = pd.to_numeric(items["item_latitude"], errors="coerce").astype(float)
    items["lon"] = pd.to_numeric(items["item_longitude"], errors="coerce").astype(float)
    item_ids = items["item_id"].to_numpy()
    tr = lb.load_train_full()
    mask = load_validation()[2]
    regions = region_location_ids()
    sparse = build_sparse_index(items, cfg)
    emb = load_corpus_embeddings(items, cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    D = torch.from_numpy(emb).to(dev, dt)
    del emb
    Dft_name, Dft = None, None

    for s in splits:
        path = OUT / f"cands_{s}.parquet"
        if path.exists():
            print(f"{s}: уже есть {path.name}")
            continue
        name = model_for(s)
        if name != Dft_name:                       # корпус моделью набора (у фолдов пула — своя)
            Dft = None
            gc.collect()
            e, ids = load_corpus_emb(f"{name}_s256")
            assert (ids.astype(str) == item_ids).all(), f"порядок item_id в {name}_s256 не совпадает с корпусом"
            Dft, Dft_name = torch.from_numpy(e).to(dev, dt), name
            del e
        q = queries(s)
        q_emb = np.load(LB_OUT / f"qemb_{s}.npy")[:len(q)]
        q_ft = qft[s][:len(q)]
        assert len(q_emb) == len(q) == len(q_ft)
        maps = build_geo_maps(items, tr, lb.train_fit_for(s, tr, mask), regions)
        t1 = time.time()
        c = union_candidates(q, q_emb, q_ft, sparse, D, Dft, maps, item_ids, cfg, tag=s)
        c.to_parquet(path, index=False)
        n_per = c.groupby("query_id").size()
        print(f"{s}: {len(q)} запросов, {len(c)} строк ({n_per.mean():.1f} на запрос, новых "
              f"{c['is_new'].mean():.3f}) за {time.time() - t1:.0f} с; модель {name}", flush=True)
        # сверка: топ-300 geo2 должен совпасть с кандидатами ltr_big
        ref = LB_OUT / f"cands_{s}.parquet"
        if ref.exists():
            r = pd.read_parquet(ref, columns=["query_id", "item_id", "rank"])
            r = r[r["query_id"].isin(set(q["query_id"]))]
            h = c[c["is_new"] == 0][["query_id", "item_id", "rank"]]
            same = h.merge(r, on=["query_id", "item_id", "rank"]).shape[0] / max(len(r), 1)
            print(f"  сверка топ-300 с ltr_big/{ref.name}: совпадает {same:.4f} строк (id и ранг)", flush=True)
        del maps
        gc.collect()
    print(f"готово за {time.time() - t0:.0f} с")


# ====================================================================== шаг 2: признаки

def ft_features(f: pd.DataFrame) -> pd.DataFrame:
    """Признаки дообученной модели: косинус, глобальный ранг (log), место в её топ-K_FT (log),
    внутри-запросные нормировки (перцентиль, разница с максимумом, z)."""
    from ltr import _in_query
    f["ft_cos_grank"] = np.log1p(f["ft_cos_grank"]).astype(np.float32)
    f["ft_rank"] = np.log1p(f["ft_rank"]).astype(np.float32)
    f["ft_cos_pct"] = _in_query(f, "ft_cos", "pct")
    f["ft_cos_gap"] = _in_query(f, "ft_cos", "gap")
    f["ft_cos_z"] = _in_query(f, "ft_cos", "z")
    return f


def features_for(cands: pd.DataFrame, q: pd.DataFrame, items: pd.DataFrame, mc, geo,
                 labels: dict | None = None) -> pd.DataFrame:
    """Признаки кандидатов union_candidates: build_features (ltr.py) + extra_features (ltr_big) + FT_FEATS.
    mc / geo — классификатор микрокатегорий и GeoCtx по train_fit набора. По частям по 1000 запросов
    (все признаки внутри-запросные). labels -> колонка y. Признаки хранятся float32."""
    from ltr import FEATURES, build_features
    keep = ["query_id", "item_id", "rank"] + FEATURES + lb.EXTRA + FT_FEATS + (["y"] if labels is not None else [])
    parts = []
    for c0 in range(0, len(q), 1000):
        qc = q.iloc[c0:c0 + 1000]
        c = cands[cands["query_id"].isin(set(qc["query_id"]))]
        src = c[["query_id", "item_id", "bm25", "bm25_grank", "cos", "cos_grank"]]
        f = build_features(c[["query_id", "item_id", "score", "rank"]], qc, items, src, mc, geo)
        f = lb.extra_features(f, qc, items)
        fx = c[["query_id", "item_id", "ft_cos", "ft_cos_grank", "ft_rank", "is_new"]]
        f = f.merge(fx, on=["query_id", "item_id"], how="left", validate="1:1")
        assert f["ft_cos"].notna().all()
        f = ft_features(f)
        if labels is not None:
            f["y"] = np.fromiter((i in labels[qq] for qq, i in zip(f["query_id"], f["item_id"])), bool,
                                 len(f)).astype(np.int8)
        f = f[keep].copy()
        for col in FEATURES + lb.EXTRA + FT_FEATS:
            f[col] = f[col].astype(np.float32)
        parts.append(f)
        del c, src
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def cmd_features(splits: list[str]):
    """Как ltr_big.cmd_features (объекты по train_fit набора), плюс FT_FEATS."""
    import rerank_meta as rm
    from geo import centroids
    from ltr import GeoCtx, _load_items
    from validation import load_validation, region_location_ids
    OUT = out_dir()
    t0 = time.time()
    items = _load_items()
    tr = lb.load_train_full()
    vq, vlabels, mask = load_validation()
    regions = region_location_ids()
    corpus = items[["item_id", "item_location_id"]].copy()
    corpus["lat"] = pd.to_numeric(items["item_latitude"], errors="coerce").astype(float)
    corpus["lon"] = pd.to_numeric(items["item_longitude"], errors="coerce").astype(float)
    cent = centroids(corpus, tr)
    tr = tr[["search_query", "search_infm_params_text", "item_microcat_id", "search_location_id",
             "item_location_id"]].copy()
    gc.collect()
    pool_labels = lb.load_pool()[1]
    for s in splits:
        path = OUT / f"feat_{s}.parquet"
        if path.exists():
            print(f"{s}: уже есть {path.name}")
            continue
        t1 = time.time()
        fit = lb.train_fit_for(s, tr, mask)
        mc = rm.fit_microcat(fit)
        geo = GeoCtx(fit, cent, regions)
        labels = vlabels if s == "val" else (pool_labels if s.startswith("tr") else None)
        f = features_for(pd.read_parquet(OUT / f"cands_{s}.parquet"), queries(s), items, mc, geo, labels)
        f.to_parquet(path, index=False)
        print(f"{s}: {len(f)} строк, признаки за {time.time() - t1:.0f} с (всего {time.time() - t0:.0f} с)",
              flush=True)
        del f, mc, geo, fit
        gc.collect()


# ====================================================================== проверка утечки

def cmd_leak():
    """Если бы модель фолда видела строки своих запросов, у пула косинус правильного ответа и его место
    в списке дообученной модели были бы заметно лучше, чем у валидации в сопоставимом слое.
    Опора — готовая bge-m3 (cos): она не дообучалась, её уровень показывает, насколько слои сами по себе
    отличаются; смотрим на прирост дообученной модели к ней (ft − bge) у пула и у валидации."""
    from validation import load_validation
    OUT = out_dir()
    RES = OUT / "res"
    RES.mkdir(exist_ok=True)
    vq, vlabels, _ = load_validation()
    pq, plabels = lb.load_pool()
    cols = ["query_id", "item_id", "rank", "y", "cos", "ft_cos", "ft_rank", "cos_grank", "ft_cos_grank", "is_new"]
    ft = pd.concat([pd.read_parquet(OUT / f"feat_{s}.parquet", columns=cols) for s in TRAIN_SETS
                    if (OUT / f"feat_{s}.parquet").exists()], ignore_index=True)
    fv = pd.read_parquet(OUT / "feat_val.parquet", columns=cols)
    one = pq.sample(frac=1.0, random_state=3).drop_duplicates("text_norm")      # по одному запросу на текст
    groups_ = {
        "пул: seen, 1 на текст": (ft, one[one["mode"] == "seen"], plabels),
        "валидация: текст в train": (fv, vq[vq["text_in_train"]], vlabels),
        "пул: unseen (nat+forced), 1 на текст": (ft, one[one["mode"] != "seen"], plabels),
        "валидация: текста нет в train": (fv, vq[~vq["text_in_train"]], vlabels),
        "пул: все": (ft, pq, plabels),
        "валидация: все": (fv, vq, vlabels),
    }
    rows = []
    for nm, (f, g, lab) in groups_.items():
        ids = set(g["query_id"])
        ff = f[f["query_id"].isin(ids)]
        if not len(ff):
            continue
        pos = ff[ff["y"] == 1]
        n_rel = pd.Series({x: len(lab[x]) for x in ff["query_id"].unique()})
        at_all = ff.groupby("query_id")["y"].sum().reindex(n_rel.index) / n_rel
        at_300 = ff[ff["is_new"] == 0].groupby("query_id")["y"].sum().reindex(n_rel.index).fillna(0) / n_rel
        ft_top = ff[ff["ft_rank"] <= np.log1p(K_FT) + 1e-6]
        at_ft = ft_top.groupby("query_id")["y"].sum().reindex(n_rel.index).fillna(0) / n_rel
        rows.append({"группа": nm, "запросов": ff["query_id"].nunique(),
                     "R@300 geo2": at_300.mean(), f"R ft топ-{K_FT}": at_ft.mean(), "R@объединения": at_all.mean(),
                     "cos bge правильного": pos["cos"].mean(), "ft_cos правильного": pos["ft_cos"].mean(),
                     "ft − bge": (pos["ft_cos"] - pos["cos"]).mean(),
                     "глоб. ранг bge (медиана)": np.expm1(pos["cos_grank"]).median(),
                     "глоб. ранг ft (медиана)": np.expm1(pos["ft_cos_grank"]).median()})
    d = pd.DataFrame(rows)
    print(d.round(4).to_string(index=False), flush=True)
    d.to_csv(RES / "leak_ft.csv", index=False)


# ====================================================================== шаг 3: модели

def load_split(split: str, cols: list[str] | None = None) -> pd.DataFrame:
    """Признаки набора из feat_<split>.parquet."""
    return pd.read_parquet(out_dir() / f"feat_{split}.parquet", columns=cols)


def cmd_final(variants: list[str], n_models: int):
    """Ансамбль как ltr_big.cmd_final (те же параметры LGBM, веса слоёв, ранняя остановка по своим 10% текстов
    пула у каждой модели, среднее z-оценок). Варианты (заданы заранее, не подбираются по валидации):
        main   — объединённые кандидаты, признаки ltr_big + FT_FEATS  (итоговый файл hybrid_ltrft_*);
        noext  — только топ-300 geo2 (без добора), признаки ltr_big + FT_FEATS: вклад одного признака;
        noft   — объединённые кандидаты, признаки ltr_big без FT_FEATS: вклад одного добора (контроль)."""
    from ltr import FEATURES, to_cands
    from validation import load_validation
    OUT = out_dir()
    RES = OUT / "res"
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    vq, vlabels, _ = load_validation()
    pq, plabels = lb.load_pool()
    bq = lb.queries_of("bench")
    base_feats = FEATURES + lb.EXTRA
    feats_of = {"main": base_feats + FT_FEATS, "noext": base_feats + FT_FEATS, "noft": base_feats}
    cols = ["query_id", "item_id", "rank", "y"] + base_feats + FT_FEATS
    ft_all = pd.concat([load_split(s, cols) for s in TRAIN_SETS], ignore_index=True)
    fv_all, fb_all = load_split("val"), load_split("bench")
    print(f"пул: {len(ft_all)} строк, {ft_all['query_id'].nunique()} запросов; {time.time() - t0:.0f} с", flush=True)
    texts = pq["text_norm"].drop_duplicates().to_numpy()
    perm = np.random.default_rng(10).permutation(len(texts))
    qids = pd.Index(sorted(fv_all["query_id"].unique()))           # все 3000 (меньше только при --limit)
    # база для сравнения: LTR-big (ltr_big.py final), если его файл есть; иначе geo2 — порядок кандидатов до LTR
    if (CANDS / "hybrid_geo2_ltrbig_val.parquet").exists():
        r_ltrbig = lb.cands_recall(CANDS / "hybrid_geo2_ltrbig_val.parquet", vlabels).reindex(qids)
    else:
        print("нет work/cands/hybrid_geo2_ltrbig_val.parquet: сравнение с geo2 вместо LTR-big", flush=True)
        g = fv_all[fv_all["is_new"] == 0]
        r_ltrbig = lb.recall_of(g, -g["rank"].to_numpy(float), vlabels).reindex(qids)
    meta_path = CANDS / "hybrid_geo2_meta_val.parquet"
    r_man = lb.cands_recall(meta_path, vlabels).reindex(qids) if meta_path.exists() else None
    d = vq.set_index("query_id").loc[qids]
    nh = ~d["target_has_history"].to_numpy()
    SL = {"все": np.ones(len(d), bool), "с фильтром": d.has_filter.to_numpy(), "без фильтра": ~d.has_filter.to_numpy(),
          "текст есть в train": d.text_in_train.to_numpy(), "текста нет в train": ~d.text_in_train.to_numpy(),
          "поиск по городу": ~d.is_region.to_numpy(), "регион или вся Россия": d.is_region.to_numpy(),
          "у ответа есть история": ~nh, "ответ без истории": nh}
    # половина A/B, как в 11/16/18: проверка, что прирост держится на отложенной половине
    perm_ab = np.random.default_rng(0).permutation(len(qids))
    isB = np.zeros(len(qids), bool)
    isB[perm_ab[len(qids) // 2:]] = True
    rows, per_q = [], {"LTR-big (0.901)": r_ltrbig}
    if r_man is not None:
        per_q["geo2+бонусы"] = r_man
    for var in variants:
        feats = feats_of[var]
        if var == "noext":
            ft, fv, fb = (x[x["is_new"] == 0].reset_index(drop=True) for x in (ft_all, fv_all, fb_all))
        else:
            ft, fv, fb = ft_all, fv_all, fb_all
        has_pos = ft.groupby("query_id", sort=False)["y"].transform("max").to_numpy() > 0
        pv, pb, per_model = np.zeros(len(fv)), np.zeros(len(fb)), []
        for k in range(n_models):
            es_texts = set(texts[perm[k::10]])
            es_ids = set(pq.loc[pq["text_norm"].isin(es_texts), "query_id"])
            is_es = ft["query_id"].isin(es_ids).to_numpy()
            sub = pq[~pq["query_id"].isin(es_ids)]
            w = lb.pool_weights(sub, bq)
            m = lb.fit_ranker(ft[~is_es & has_pos], ft[is_es & has_pos], feats, w, seed=k)
            a = m.predict(fv[feats].to_numpy(np.float32))
            pv += lb._zq(fv, a)
            pb += lb._zq(fb, m.predict(fb[feats].to_numpy(np.float32)))
            per_model.append(lb.recall_of(fv, a, vlabels).mean())
            if var == "main":                    # модели итоговой схемы — для src/pipeline.py (--ltr)
                (OUT / "models").mkdir(parents=True, exist_ok=True)
                m.booster_.save_model(str(OUT / "models" / f"ltr_{k}.txt"), num_iteration=m.best_iteration_)
            print(f"  {var} модель {k}: деревьев {m.best_iteration_}, val {per_model[-1]:.4f}; "
                  f"{time.time() - t0:.0f} с", flush=True)
            if k == 0:
                imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
                imp.to_csv(RES / f"importance_{var}.csv")
                print("  важность (gain), топ-12: " + ", ".join(f"{i} {v / imp.sum():.3f}" for i, v in imp.head(12).items()))
        pv /= n_models
        pb /= n_models
        if var == "main":
            (OUT / "models" / "ltr.json").write_text(json.dumps(
                {"features": feats, "n_models": n_models, "ft_model": ARGS.val_model, "k_ft": K_FT, "n_fused": N_FUSED,
                 "val_recall50": float(lb.recall_of(fv, pv, vlabels).mean())}, ensure_ascii=False, indent=1))
        r = lb.recall_of(fv, pv, vlabels).reindex(qids)
        per_q[var] = r
        dd, dB, dn = lb.boot(r, r_ltrbig), lb.boot(r[isB], r_ltrbig[isB]), lb.boot(r[nh], r_ltrbig[nh])
        rows.append(dict(вариант=var, R50=r.mean(), модели=str(np.round(per_model, 4)), без_истории=r[nh].mean(),
                         d_ltrbig=dd[0], lo=dd[1], hi=dd[2], dB=dB[0], B_lo=dB[1], B_hi=dB[2],
                         d_nh=dn[0], nh_lo=dn[1], nh_hi=dn[2]))
        print(f"{var}: val {r.mean():.4f} (без истории {r[nh].mean():.4f}); Δ к LTR-big {dd[0]:+.4f} "
              f"[{dd[1]:+.4f};{dd[2]:+.4f}], на B {dB[0]:+.4f} [{dB[1]:+.4f};{dB[2]:+.4f}], без истории {dn[0]:+.4f} "
              f"[{dn[1]:+.4f};{dn[2]:+.4f}]", flush=True)
        pd.DataFrame(rows).to_csv(RES / "final.csv", index=False)
        cv_, cb = to_cands(fv, pv), to_cands(fb, pb)
        cv_ = cv_[cv_["rank"] <= 300].reset_index(drop=True)
        cb = cb[cb["rank"] <= 300].reset_index(drop=True)
        assert cv_.groupby("query_id").size().eq(300).all() and cv_["query_id"].nunique() == len(qids)
        assert cb.groupby("query_id").size().eq(300).all() and cb["query_id"].nunique() == fb["query_id"].nunique()
        dst = (CANDS if var == "main" and Path(ARGS.out) == WORK / "ltr_ft" else OUT)
        sfx = "" if var == "main" else f"_{var}"
        cv_.to_parquet(dst / f"hybrid_ltrft{sfx}_val.parquet", index=False)
        cb.to_parquet(dst / f"hybrid_ltrft{sfx}_bench.parquet", index=False)
        chk = lb.cands_recall(dst / f"hybrid_ltrft{sfx}_val.parquet", {x: vlabels[x] for x in qids})
        assert abs(chk.mean() - r.mean()) < 1e-9, (chk.mean(), r.mean())
        top = cb[cb["rank"] <= 50]
        msg = f"  файл {dst.name}/hybrid_ltrft{sfx}_*: val R@50 {chk.mean():.4f}; новых кандидатов в топ-50 бенчмарка " \
              f"{top.merge(fb[['query_id', 'item_id', 'is_new']], on=['query_id', 'item_id'])['is_new'].mean():.3f}"
        if (CANDS / "hybrid_geo2_ltrbig_bench.parquet").exists():
            o = pd.read_parquet(CANDS / "hybrid_geo2_ltrbig_bench.parquet")
            o = o[o["rank"] <= 50]
            msg += f"; топ-50 бенчмарка совпадает с LTR-big на {top.merge(o, on=['query_id', 'item_id']).shape[0] / len(top):.3f}"
        print(msg, flush=True)
        gc.collect()
    sl = pd.DataFrame({k: {"n": int(mk.sum()), **{nm: r[mk].mean() for nm, r in per_q.items()}}
                       for k, mk in SL.items()}).T
    print(sl.round(4).to_string())
    sl.to_csv(RES / "slices.csv")
    print(f"готово за {time.time() - t0:.0f} с")


def encode_ft_queries(name: str, q: pd.DataFrame) -> np.ndarray:
    """Эмбеддинги запросов дообученной моделью work/models/<name> — тем же путём, что ltr_big.ft_query_emb
    (веса сразу в fp16, запрос = текст + значения фильтров, префикс из train_info.json), без кэша."""
    import torch
    from sentence_transformers import SentenceTransformer
    from ft_eval import MODELS_DIR, encode_q, prefixes
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"dtype": torch.float16} if dev == "cuda" else {}
    m = SentenceTransformer(str(MODELS_DIR / name), device="cpu", model_kwargs=kw).to(dev)
    m.max_seq_length = 256
    e = encode_q(m, q, prefixes(name)[0])
    del m
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return e


def ltr_predict(f: pd.DataFrame, models_dir: Path = MODELS_OUT) -> np.ndarray:
    """Скор итоговой схемы: среднее z-оценок (внутри запроса) предсказаний 5 сохранённых бустеров, как в cmd_final."""
    import lightgbm as lgb
    meta = json.loads((models_dir / "ltr.json").read_text())
    X = f[meta["features"]].to_numpy(np.float32)
    p = np.zeros(len(f))
    for k in range(meta["n_models"]):
        b = lgb.Booster(model_file=str(models_dir / f"ltr_{k}.txt"))
        p += lb._zq(f, b.predict(X))
    return p / meta["n_models"]


def main():
    """Разбор аргументов командной строки и запуск шага."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["retrieve", "features", "leak", "final"])
    ap.add_argument("--splits", default="val,bench," + ",".join(TRAIN_SETS))
    ap.add_argument("--variants", default="main,noext,noft")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--out", default=str(WORK / "ltr_ft"))
    ap.add_argument("--fold-model", default=FT_FOLD, help="шаблон имени модели фолда, {f} = номер")
    ap.add_argument("--val-model", default=FT_VAL, help="дообученная модель для валидации и бенчмарка")
    ap.add_argument("--limit", type=int, default=0, help="только первые N запросов набора (проверка кода)")
    a = ap.parse_args()
    ARGS.out, ARGS.fold_model, ARGS.val_model, ARGS.limit = Path(a.out), a.fold_model, a.val_model, a.limit
    splits = a.splits.split(",")
    if a.cmd == "retrieve":
        cmd_retrieve(splits)
    elif a.cmd == "features":
        cmd_features(splits)
    elif a.cmd == "leak":
        cmd_leak()
    else:
        cmd_final(a.variants.split(","), a.n_models)


if __name__ == "__main__":
    main()
