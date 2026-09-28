"""Qwen3-Embedding-0.6B третьим источником в новой гео-схеме geo2 (+ бонусы метаданных), см. eda/13_qwen_ensemble.md.

Обёртка над src/geo_run.py и src/geo.py (они не меняются): та же конфигурация, что у
    python src/geo_run.py final --sparse lemma --region RSOFT_0.02 --russia RSOFT_0.005 --city SOFT_0.015
только в RRF идут три списка по ярусам: BM25 (lemma+stem), bge-m3, Qwen3. За один проход по запросам
считаются варианты:
    2src      — BM25 + bge-m3 (контроль, должен воспроизвести hybrid_geo2_val = 0.858);
    +qwen_g1   — + Qwen3, гео-бонусы прежние;
    +qwen_g1.5 — + Qwen3, гео-бонусы ×1.5 (у трёх источников RRF-скор примерно в 1.5 раза крупнее);
    +e5_g1 / +e5_g1.5 — то же с e5-small seq 256 третьим (корпус кодируется за 2.5 мин — контроль цены).
Потом бонусы метаданных (rerank_meta, веса W0 × s, base="score"). Вариант (гео-множитель, s) выбирается на
половине A валидации (seed 0, как в rerank_meta_geo2.py), результат — на отложенной половине B.

Эмбеддинги запросов берутся из кеша work/emb/*_q_{val,bench}.npy (Qwen3 пишет experiments/dense_qwen.py run;
bge-m3 и e5-small кодируются здесь при отсутствии кеша). На GPU одновременно только матрицы корпуса трёх моделей (~0.9 ГБ).

Запуск из корня репозитория: python experiments/qwen_geo2.py   → work/cands/hybrid3_{val,bench}.parquet (+ Qwen3),
                                                  work/cands/hybrid3_e5_{val,bench}.parquet (+ e5-small, контроль)
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geo import RUSSIA, rrf, tiered_top  # noqa: E402
from geo_run import build_geo, build_sparse, save_cands  # noqa: E402
from sparse import SparseIndex, load_items  # noqa: E402
from validation import load_validation, recall_at_k, region_location_ids, slices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, CANDS, EMB = ROOT / "data", ROOT / "work" / "cands", ROOT / "work" / "emb"
K = 300
RULES = dict(region="RSOFT_0.02", russia="RSOFT_0.005", city="SOFT_0.015")
# источники: 0 BM25, 1 bge-m3, 2 Qwen3, 3 e5-small (дешёвый контроль третьего dense-источника)
VARIANTS = {"2src": ((0, 1), 1.0),
            "+qwen_g1": ((0, 1, 2), 1.0), "+qwen_g1.5": ((0, 1, 2), 1.5),
            "+e5_g1": ((0, 1, 3), 1.0), "+e5_g1.5": ((0, 1, 3), 1.5)}  # имя -> (источники, множитель гео-бонуса)
META_SCALES = (1.0, 1.5, 2.0)
W0 = dict(w_vid=0.003, w_tip=0.0015, w_mc=0.007, w_prf=0.002)  # веса из eda/11_meta_boost.md
TMP = Path(tempfile.mkdtemp(prefix="qwen_geo2_"))  # save_cands пишет файл — промежуточные копии во временный каталог
QWEN_TAG, BGE_TAG, E5_TAG = "qwen3-0.6b_s192", "bge-m3_s256", "e5-small_s256"


def st_queries(key: str, tag: str, queries: pd.DataFrame, split: str) -> np.ndarray:
    """Эмбеддинги запросов модели из dense.MODELS (с фильтрами, как в geo_run) с кешем; модель выгружается сразу."""
    f = EMB / f"{tag}_q_{split}.npy"
    if f.exists():
        return np.load(f)
    from dense import MODELS, encode_queries
    from sentence_transformers import SentenceTransformer
    # как dense.load_model (fp16, seq 256), но веса переводятся в fp16 ещё на CPU:
    # загрузка сразу на GPU кладёт туда fp32-копию (~2.3 ГБ) и не влезает в лимит
    m = SentenceTransformer(MODELS[key]["name"], device="cpu").half().to("cuda")
    m.max_seq_length = 256
    e = encode_queries(m, key, queries, with_filters=True, batch_size=8)
    del m
    import gc
    gc.collect()  # иначе веса остаются на GPU (циклические ссылки)
    torch.cuda.empty_cache()
    np.save(f, e)
    return e


def rule_of(loc, regions) -> str:
    if loc == RUSSIA:
        return RULES["russia"]
    return RULES["region"] if loc in regions else RULES["city"]


def run3(queries, sp, Ds: list, Qs: list, maps, regions) -> dict:
    """Для каждого запроса — списки трёх источников по ярусам и RRF всех вариантов.
    Возвращает {вариант: {query_id: (позиции, скоры)}}."""
    n = len(queries)
    locs = queries["search_location_id"].to_numpy()
    texts = queries["search_query"].to_numpy()
    qids = queries["query_id"].to_numpy()
    kept = {v: {} for v in VARIANTS}
    t0, bs = time.time(), 128
    for b0 in range(0, n, bs):
        with torch.no_grad():
            dsc = [(torch.from_numpy(Q[b0:b0 + bs]).to(D.device, D.dtype) @ D.T).float().cpu().numpy()
                   for D, Q in zip(Ds, Qs)]
        for j in range(dsc[0].shape[0]):
            i = b0 + j
            rule = rule_of(int(locs[i]), regions)
            tiers = maps.tiers(maps.base_rule(rule), int(locs[i]), maps.city.loc(texts[i]))
            ssc = sp.scores(texts[i])
            g_s = SparseIndex._top(np.arange(len(ssc)), ssc, 2 * K)
            lists = [tiered_top(ssc, tiers, K, g_s, positive=True)[0]]
            for d in dsc:
                dv = d[j]
                g = np.argpartition(-dv, 2 * K)[:2 * K]
                g = g[np.argsort(-dv[g], kind="stable")]
                lists.append(tiered_top(dv, tiers, K, g, positive=False)[0])
            bmask, w = maps.bonus_mask(rule, int(locs[i]))
            for v, (src, gm) in VARIANTS.items():
                kept[v][qids[i]] = rrf([lists[s] for s in src], bonus=bmask, bonus_w=w * gm)
        if (b0 // bs) % 5 == 0:
            print(f"  {min(b0 + bs, n)}/{n} за {time.time() - t0:.0f} с", flush=True)
    return kept


def boot(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0):
    d = (a - b.reindex(a.index)).to_numpy()
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def rec(c: pd.DataFrame, labels: dict, k: int = 50) -> pd.Series:
    top = c[c["rank"] <= k]
    return recall_at_k(top.groupby("query_id")["item_id"].agg(list).to_dict(), labels, k)


def main():
    import rerank_meta as rm
    from rerank_meta_exp import features
    t0 = time.time()
    torch.cuda.set_per_process_memory_fraction(0.19)  # ≈1.45 ГБ: GPU делится с другими процессами
    regions = region_location_ids()
    vq, vl, mask = load_validation()
    vq["query_id"] = vq["query_id"].astype(str)
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    bq["query_id"] = bq["query_id"].astype(str)
    # запросы bge-m3 кодируются до загрузки матриц корпуса (веса модели и матрицы не лежат на GPU вместе)
    Qbge = {sp_: st_queries("bge-m3", BGE_TAG, q, sp_) for sp_, q in (("val", vq), ("bench", bq))}
    Qe5 = {sp_: st_queries("e5-small", E5_TAG, q, sp_) for sp_, q in (("val", vq), ("bench", bq))}
    items = load_items()
    corpus_ids = items["item_id"].astype(str).to_numpy()
    Ds = []
    for tag in (BGE_TAG, QWEN_TAG, E5_TAG):
        e = np.load(EMB / f"{tag}.npy")
        assert (np.load(EMB / f"{tag}_ids.npy", allow_pickle=True) == corpus_ids).all(), tag
        Ds.append(torch.from_numpy(e).to("cuda", torch.float16))
        del e
    from geo import load_geo_data
    corpus, train = load_geo_data()
    assert (corpus["item_id"].to_numpy() == corpus_ids).all()
    sp = build_sparse(items, "lemma")
    del items
    print(f"подготовка {time.time() - t0:.0f} с", flush=True)

    # ---------------- валидация
    maps = build_geo(corpus, train, train[mask], regions)
    Qv = [Qbge["val"], np.load(EMB / f"{QWEN_TAG}_q_val.npy"), Qe5["val"]]
    kept = run3(vq, sp, Ds, Qv, maps, regions)
    cands = {v: save_cands(k, vq, corpus_ids, TMP / f"_tmp_{v}.parquet") for v, k in kept.items()}
    for v in VARIANTS:
        (TMP / f"_tmp_{v}.parquet").unlink()
    print(f"валидация: {time.time() - t0:.0f} с", flush=True)

    tr = pd.read_parquet(DATA / "train.parquet", columns=["search_query", "search_infm_params_text", "item_microcat_id"])
    mitems = rm.load_items(DATA / "benchmark_items.parquet")
    mc_model = rm.fit_microcat(tr[mask])
    qids = np.array(sorted(vl))
    perm = np.random.default_rng(0).permutation(len(qids))
    half_b = set(qids[perm[len(qids) // 2:]])

    res, out = {}, {}
    for v, c in cands.items():
        res[(v, "—")] = rec(c, vl)
        res[(v, "—", 300)] = rec(c, vl, 300)
        f = features(c, vq, mitems, mc_model)
        for s in META_SCALES:
            r = rm.rerank_bonus(f, base="score", **{k: w * s for k, w in W0.items()})
            out[(v, s)] = r
            res[(v, s)] = rec(r, vl)
            res[(v, s, 300)] = rec(r, vl, 300)
        del f
    ctrl = res[("2src", 1.0)]  # = hybrid_geo2_meta_val (0.869)
    ib = ctrl.index.isin(half_b)
    rows = []
    for key, r in res.items():
        if len(key) == 3:
            continue
        d, lo, hi = boot(r, ctrl)
        dB, loB, hiB = boot(r[ib], ctrl[ib])
        rows.append(dict(вариант=key[0], мета=key[1], A=r[~ib].mean(), B=r[ib].mean(), все=r.mean(),
                         r300=res[key + (300,)].mean(), d=d, lo=lo, hi=hi, dB=dB, loB=loB, hiB=hiB))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nRecall@50 geo2 (A/B — половины валидации; Δ к 2src + мета ×1.0 = 0.869; ДИ — парный бутстреп)")
    print(tab.round(4).to_string(index=False))
    tab.to_csv(ROOT / "work" / "qwen_geo2_grid.csv", index=False)

    # выбор (гео-множитель, масштаб мета) внутри семейства «+qwen» и «+e5» — по половине A, проверка на B
    chosen = {}
    for fam, fname in (("+qwen", "hybrid3"), ("+e5", "hybrid3_e5")):
        cand3 = tab[tab["вариант"].str.startswith(fam) & (tab["мета"] != "—")]
        best = cand3.loc[cand3["A"].idxmax()]
        bv, bs = best["вариант"], float(best["мета"])
        chosen[fam] = (bv, bs, fname)
        rb = res[(bv, bs)]
        print(f"\n{fam}: выбрано по A: {bv}, мета ×{bs}")
        d = boot(rb[ib], ctrl[ib])
        print(f"  B: 0.869-схема {ctrl[ib].mean():.4f} -> {rb[ib].mean():.4f}, Δ {d[0]:+.4f} [{d[1]:+.4f}; {d[2]:+.4f}]")
        d = boot(rb, ctrl)
        print(f"  все: {ctrl.mean():.4f} -> {rb.mean():.4f}, Δ {d[0]:+.4f} [{d[1]:+.4f}; {d[2]:+.4f}]")
        out[(bv, bs)].to_parquet(CANDS / f"{fname}_val.parquet", index=False)
    rq, re5 = (res[chosen[f][:2]] for f in ("+qwen", "+e5"))
    d = boot(rq[ib], re5[ib])
    print(f"\n+qwen − +e5 на B: {d[0]:+.4f} [{d[1]:+.4f}; {d[2]:+.4f}]")
    sl = pd.concat({"geo2+мета (0.869)": slices(ctrl, vq)["recall"], "+qwen": slices(rq, vq)["recall"],
                    "+e5": slices(re5, vq)["recall"]}, axis=1)
    sl.insert(0, "запросов", slices(ctrl, vq)["запросов"].astype(int))
    print(sl.round(4).to_string())

    # ---------------- бенчмарк: карты и классификатор — по полному train
    maps_b = build_geo(corpus, train, train, regions)
    Qb = [Qbge["bench"], np.load(EMB / f"{QWEN_TAG}_q_bench.npy"), Qe5["bench"]]
    keep = {v: VARIANTS[v] for v, _, _ in chosen.values()}
    VARIANTS.clear(); VARIANTS.update(keep)
    kept_b = run3(bq, sp, Ds, Qb, maps_b, regions)
    mc_b = rm.fit_microcat(tr)
    old = pd.read_parquet(CANDS / "hybrid_geo2_meta_bench.parquet")
    for bv, bs, fname in chosen.values():
        cb = save_cands(kept_b[bv], bq, corpus_ids, TMP / "_tmp_bench.parquet")
        nb = rm.rerank_bonus(features(cb, bq, mitems, mc_b), base="score", **{k: w * bs for k, w in W0.items()})
        assert nb.groupby("query_id").size().reindex(bq["query_id"]).eq(K).all()
        nb.to_parquet(CANDS / f"{fname}_bench.parquet", index=False)
        same = nb[nb["rank"] <= 50].merge(old[old["rank"] <= 50], on=["query_id", "item_id"]).shape[0] / (50 * len(bq))
        print(f"бенчмарк {fname}: {len(bq)} запросов; топ-50 совпадает с hybrid_geo2_meta_bench на {same:.3f}")
    print(f"всего {time.time() - t0:.0f} с; пик VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} ГБ")


if __name__ == "__main__":
    main()
