"""Эксперименты с географией поиска: гибрид RRF (BM25 + bge-m3) по ярусам кандидатов из src/geo.py.

Для каждого запроса один раз считаются полные векторы скоров обоих источников по корпусу
(BM25 — index.scores, bge-m3 — матрица запросы × корпус на GPU батчами). Потом для каждого правила
из geo.RULES: ярусы -> top-300 каждого источника по ярусам с добором из всего корпуса -> RRF k=60 -> метрики.

Запуск из корня репозитория:
    python src/geo_run.py exp --sparse stem          # все правила на валидации -> work/geo/val_stem.parquet
    python src/geo_run.py exp --sparse lemma
    python src/geo_run.py final --sparse lemma --region R3 --russia R5 --city C3 [--bonus 0]
        # выбранная конфигурация: валидация (карты по train[mask]) и бенчмарк (карты по полному train);
        # пишет work/cands/hybrid_geo2_{val,bench}.parquet и answer_03_geo.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import write_answer  # noqa: E402
from dense import encode_queries, load_corpus_emb, load_model  # noqa: E402
from geo import (CITY_RULES, REGION_RULES, RUSSIA, CityFinder, GeoMaps, centroids, load_geo_data,  # noqa: E402
                 rrf, tiered_top, train_item_params)
from sparse import SparseIndex, load_items  # noqa: E402
from validation import load_validation, region_location_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "work" / "geo"
CANDS = ROOT / "work" / "cands"
K = 300
KS = (50, 100, 200, 300)


def build_sparse(items: pd.DataFrame, kind: str):
    """BM25 на стемах или на леммах+стемах (вариант документа b)."""
    if kind == "stem":
        return SparseIndex.build(items, variant="b")
    from sparse_lemma import LemmaIndex
    return LemmaIndex.build(items, scheme="lemma+stem", variant="b")


def build_geo(corpus, train, train_fit, regions):
    """Центроиды — по объявлениям корпуса + всего train (координаты — не ответы);
    словарь-фильтр городов и карты — только по train_fit."""
    cent = centroids(corpus, train)
    city = CityFinder.build(corpus, train_item_params(), cent, train_fit)
    return GeoMaps.build(corpus, cent, regions, city, train_fit)


def run(queries: pd.DataFrame, rule_of, sp, D, q_emb, maps: GeoMaps, labels_pos: dict | None,
        keep: bool = False):
    """Главный цикл. rule_of(query_row) -> список правил, которые надо посчитать для запроса.
    Возвращает (метрики по (запрос, правило), кандидаты top-300 если keep)."""
    n = len(queries)
    locs = queries["search_location_id"].to_numpy()
    texts = queries["search_query"].to_numpy()
    qids = queries["query_id"].to_numpy()
    named = [maps.city.loc(t) for t in texts]
    rows, kept = [], {}
    t0 = time.time()
    bs = 128
    for b0 in range(0, n, bs):
        with torch.no_grad():
            dsc = (torch.from_numpy(q_emb[b0:b0 + bs]).to(D.device, D.dtype) @ D.T).float().cpu().numpy()
        for j in range(dsc.shape[0]):
            i = b0 + j
            ssc = sp.scores(texts[i])
            dv = dsc[j]
            g_s = SparseIndex._top(np.arange(len(ssc)), ssc, 2 * K)          # глобальный топ, скор > 0
            g_d = np.argpartition(-dv, 2 * K)[:2 * K]
            g_d = g_d[np.argsort(-dv[g_d], kind="stable")]
            for rule in rule_of(i):
                tiers = maps.tiers(maps.base_rule(rule), int(locs[i]), named[i])
                ls, nts = tiered_top(ssc, tiers, K, g_s, positive=True)
                ld, ntd = tiered_top(dv, tiers, K, g_d, positive=False)
                bmask, w = maps.bonus_mask(rule, int(locs[i]))
                fused, fsc = rrf([ls, ld], bonus=bmask, bonus_w=w)
                if keep:
                    kept[qids[i]] = (fused, fsc)
                if labels_pos is None:
                    continue
                lab = labels_pos[qids[i]]
                pool = np.concatenate(tiers) if tiers else None
                r = {"query_id": qids[i], "rule": rule, "named": named[i] is not None,
                     "pool_size": 0 if pool is None else len(pool),
                     "in_pool": np.nan if pool is None else float(np.isin(lab, pool).mean()),
                     "sparse50": float(np.isin(lab, ls[:50]).mean()),
                     "dense50": float(np.isin(lab, ld[:50]).mean())}
                for k in KS:
                    r[f"r{k}"] = float(np.isin(lab, fused[:k]).mean())
                rows.append(r)
        if (b0 // bs) % 5 == 0:
            print(f"  {min(b0 + bs, n)}/{n} за {time.time() - t0:.0f} с", flush=True)
    return pd.DataFrame(rows), kept


def prepare(sparse_kind: str):
    """Корпус, эмбеддинги bge-m3, гео-данные и BM25 для прогонов гео-правил."""
    items = load_items()
    emb, ids = load_corpus_emb("bge-m3_s256")
    assert (items["item_id"].astype(str).to_numpy() == ids).all(), "порядок item_id не совпадает с эмбеддингами"
    corpus, train = load_geo_data()
    assert (corpus["item_id"].to_numpy() == ids).all()
    t = time.time()
    sp = build_sparse(items, sparse_kind)
    print(f"sparse {sparse_kind}: индекс за {time.time() - t:.0f} с", flush=True)
    del items
    D = torch.from_numpy(emb).to("cuda", torch.float16)
    del emb
    return ids, corpus, train, sp, D


def main():
    """Разбор аргументов: exp — перебор гео-правил, final — кандидаты выбранной схемы."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["exp", "final"])
    ap.add_argument("--sparse", choices=["stem", "lemma"], default="stem")
    ap.add_argument("--region", default="R0")
    ap.add_argument("--russia", default="R0")
    ap.add_argument("--city", default="C0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ids, corpus, train, sp, D = prepare(a.sparse)
    pos_of = {x: i for i, x in enumerate(ids)}
    regions = region_location_ids()
    vq, vl, mask = load_validation()
    vq["query_id"] = vq["query_id"].astype(str)
    labels_pos = {q: np.array([pos_of[x] for x in v]) for q, v in vl.items()}
    maps = build_geo(corpus, train, train[mask], regions)
    print(f"карты (train[mask]) за {time.time() - t0:.0f} с; регионов в карте: {len(maps.region_map)}, "
          f"городов с парами: {len(maps.pairs)}, названий городов: {len(maps.city.city2loc)}, "
          f"выброшено негеографичных: {len(maps.city.bad)}", flush=True)
    model = load_model("bge-m3", 256)
    qv = encode_queries(model, "bge-m3", vq, with_filters=True)

    is_reg = vq["search_location_id"].isin(regions).to_numpy()

    if a.cmd == "exp":
        del model
        torch.cuda.empty_cache()
        city_rules = ["C0R0"] + CITY_RULES
        reg_rules = ["C0R0"] + REGION_RULES
        res, _ = run(vq, lambda i: reg_rules if is_reg[i] else city_rules, sp, D, qv, maps, labels_pos)
        res.to_parquet(OUT / f"val_{a.sparse}.parquet", index=False)
        b = res[res["rule"] == "C0R0"]
        print(f"база C0R0 ({a.sparse}): R@50 = {b['r50'].mean():.4f}; всего {time.time() - t0:.0f} с")
        return

    # ---------------- final: выбранная конфигурация
    def rule_for(locs):
        """Функция «индекс запроса -> список гео-правил» по типу локации (город / регион / вся Россия)."""
        def f(i):
            l = locs[i]
            if l == RUSSIA:
                return [a.russia if a.russia != "R0" else "C0R0"]
            if l in regions:
                return [a.region if a.region != "R0" else "C0R0"]
            return [a.city if a.city != "C0" else "C0R0"]
        return f

    vlocs = vq["search_location_id"].to_numpy()
    res, kept = run(vq, rule_for(vlocs), sp, D, qv, maps, labels_pos, keep=True)
    res.to_parquet(OUT / f"final_val_{a.sparse}.parquet", index=False)
    print(f"валидация: R@50 = {res['r50'].mean():.4f}", flush=True)
    save_cands(kept, vq, ids, CANDS / "hybrid_geo2_val.parquet")

    # бенчмарк: карты и фильтр названий — по полному train
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    bq["query_id"] = bq["query_id"].astype(str)
    maps_b = build_geo(corpus, train, train, regions)
    qb = encode_queries(model, "bge-m3", bq, with_filters=True)
    del model
    torch.cuda.empty_cache()
    blocs = bq["search_location_id"].to_numpy()
    _, kept_b = run(bq, rule_for(blocs), sp, D, qb, maps_b, None, keep=True)
    cb = save_cands(kept_b, bq, ids, CANDS / "hybrid_geo2_bench.parquet")
    assert cb.groupby("query_id").size().reindex(bq["query_id"]).eq(K).all()
    preds = {q: g.tolist() for q, g in cb[cb["rank"] <= 50].groupby("query_id", sort=False)["item_id"]}
    write_answer(preds, ROOT / "answer_03_geo.csv", bq)
    named_b = np.mean([maps_b.city.loc(t) is not None for t in bq["search_query"]])
    print(f"бенчмарк: {len(bq)} запросов, с городом в тексте {named_b:.1%}; "
          f"записано answer_03_geo.csv; всего {time.time() - t0:.0f} с")


def save_cands(kept: dict, queries: pd.DataFrame, ids: np.ndarray, path: Path) -> pd.DataFrame:
    """query_id, item_id (str), score (float32, RRF), rank (int32 с 1) — топ-300 в порядке queries."""
    qs, it, sc, rk = [], [], [], []
    for q in queries["query_id"]:
        p, s = kept[q]
        qs.append(np.full(len(p), q, dtype=object)); it.append(ids[p]); sc.append(s)
        rk.append(np.arange(1, len(p) + 1))
    df = pd.DataFrame({"query_id": np.concatenate(qs), "item_id": np.concatenate(it).astype(str),
                       "score": np.concatenate(sc).astype(np.float32), "rank": np.concatenate(rk).astype(np.int32)})
    df.to_parquet(path, index=False)
    return df


if __name__ == "__main__":
    main()
