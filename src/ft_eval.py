"""Оценка дообученной модели эмбеддингов (src/ft_train.py): dense отдельно и в итоговом пайплайне.

Тонкая обёртка над geo_run.py / geo.py / rerank_meta.py (их код не меняется, только импортируется).

Команды (из корня репозитория):
    python src/ft_eval.py encode --name e5-small-ft [--batch 256]
        корпус -> work/emb/<name>_s256.{npy,_ids.npy,json} (seq 256, fp16, префикс "passage: ")
    python src/ft_eval.py dense --name e5-small-ft --ref e5-small_s256 [--ref bge-m3_s256]
        dense-only Recall@50/@300 all и geo на валидации, парный бутстреп к готовым моделям
    python src/ft_eval.py pipe --name e5-small-ft [--tag T] [--variants ...]
        итоговый пайплайн geo2 (правила lemma, RSOFT_0.02 / RSOFT_0.005 / SOFT_0.015) + бонусы метаданных
        (base=score, Вид 0.003, Тип 0.0015, P(mc) 0.007, PRF 0.002) для вариантов источников:
          bge      — BM25 + bge-m3 (воспроизводит 0.8581 / 0.8690),
          ft       — BM25 + дообученный dense вместо bge-m3,
          bge+ft   — три источника в RRF (веса 1/1/1), bge+ft_w — веса 1/0.5/0.5 (задано заранее);
        сравнение с 0.869 парным бутстрепом на всех 3000 и на половине B (seed 0, как в 11_meta_boost.md),
        срезы. Несколько --name: добавляются варианты-пары дообученных моделей (<имя1>+<имя2>_w).
        Префиксы запроса/документа берутся из work/models/<имя>/train_info.json (у bge-m3 пустые).
    python src/ft_eval.py final --name e5-small-ft --best bge+e5-small-ft_w --bench-name e5-small-ft-full
        кандидаты выбранного варианта -> work/cands/hybrid_ft_{val,bench}.parquet (топ-300, после бонусов)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_meta as rm  # noqa: E402
from dense import encode_queries, load_corpus_emb, load_model, search, to_preds  # noqa: E402
from geo import RUSSIA, rrf, tiered_top  # noqa: E402
from geo_run import K, build_geo, prepare, save_cands  # noqa: E402
from rerank_meta_exp import boot, features, recall_series  # noqa: E402
from sparse import SparseIndex  # noqa: E402
from text import item_text  # noqa: E402
from validation import load_validation, recall_at_k, region_location_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, WORK = ROOT / "data", ROOT / "work"
EMB, CANDS, MODELS_DIR, FT = WORK / "emb", WORK / "cands", WORK / "models", WORK / "ft"
# итоговая гео-конфигурация (eda/10_geo.md) и бонусы метаданных (eda/11_meta_boost.md)
GEO = dict(region="RSOFT_0.02", russia="RSOFT_0.005", city="SOFT_0.015")
W_META = dict(w_vid=0.003, w_tip=0.0015, w_mc=0.007, w_prf=0.002)
Q_PREFIX, D_PREFIX = "query: ", "passage: "


def prefixes(name: str) -> tuple[str, str]:
    """Префиксы запроса/документа, с которыми модель дообучалась (train_info.json; у bge-m3 их нет)."""
    f = MODELS_DIR / name / "train_info.json"
    if not f.exists():
        return Q_PREFIX, D_PREFIX
    args = json.loads(f.read_text())["args"]
    return args["q_prefix"], args["d_prefix"]


def load_ft(name: str, max_seq: int = 256):
    """Дообученная модель из work/models/<name> в fp16 на GPU с заданной длиной последовательности."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(MODELS_DIR / name), device="cuda").half()
    m.max_seq_length = max_seq
    return m


def encode_q(model, queries: pd.DataFrame, qp: str = Q_PREFIX) -> np.ndarray:
    """Запрос как в dense.encode_queries(with_filters=True) с префиксом модели."""
    from dense import query_text
    texts = [qp + t for t in query_text(queries, with_filters=True)]
    return model.encode(texts, batch_size=256, normalize_embeddings=True, prompt="",
                        convert_to_numpy=True).astype(np.float32)


# ------------------------------------------------------------------ encode

def cmd_encode(a):
    """Эмбеддинги корпуса дообученной моделью (item_text, 256 токенов, fp16) -> work/emb/<name>_s256.{npy,_ids.npy,json}."""
    tag = f"{a.name}_s256"
    items = pd.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    qp, dp = prefixes(a.name)
    docs = (dp + item_text(items)).tolist()
    ids = items["item_id"].astype(str).to_numpy()
    del items
    model = load_ft(a.name, 256)
    torch.cuda.synchronize()
    t0 = time.time()
    # сортировка по длине ускоряет кодирование (меньше паддинга); порядок потом восстанавливается
    order = np.argsort([-len(d) for d in docs], kind="stable")
    emb = model.encode([docs[i] for i in order], batch_size=a.batch, normalize_embeddings=True, prompt="",
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    out = np.empty_like(emb)
    out[order] = emb
    sec = time.time() - t0
    EMB.mkdir(parents=True, exist_ok=True)
    np.save(EMB / f"{tag}.npy", out)
    np.save(EMB / f"{tag}_ids.npy", ids)
    meta = dict(model=f"work/models/{a.name}", max_seq=256, device="cuda", batch=a.batch,
                gpu=torch.cuda.get_device_name(0), q_prefix=qp, d_prefix=dp, precision="fp16",
                n=len(ids), dim=int(out.shape[1]), seconds_this_run=round(sec, 1))
    (EMB / f"{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"корпус {len(ids)} за {sec:.0f} с -> {EMB / tag}.npy", flush=True)


# ------------------------------------------------------------------ dense-only

def slice_table(rs: dict, vq: pd.DataFrame) -> pd.DataFrame:
    """Recall по срезам для словаря {имя: per-query Series}."""
    d = vq.set_index("query_id")
    rows = {}
    for name, r in rs.items():
        q = d.loc[r.index]
        rows[name] = {"все": r.mean(), "текст есть в train": r[q.text_in_train.values].mean(),
                      "текста нет в train": r[~q.text_in_train.values].mean(),
                      "поиск по городу": r[~q.is_region.values].mean(),
                      "регион или вся Россия": r[q.is_region.values].mean(),
                      "с фильтром": r[q.has_filter.values].mean(), "без фильтра": r[~q.has_filter.values].mean(),
                      "у ответа есть история": r[q.target_has_history.values].mean(),
                      "ответ без истории": r[~q.target_has_history.values].mean()}
    return pd.DataFrame(rows)


def cmd_dense(a):
    """Dense-only Recall@50/@300 дообученной модели на валидации (весь корпус и гео-пул) против готовых моделей."""
    vq, labels, _ = load_validation()
    regions = region_location_ids()
    item_locs = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_location_id"])[
        "item_location_id"].to_numpy()
    res, per = [], {}
    tags = [(f"{a.name}_s256", "ft")] + [(r, "ref") for r in a.ref]
    for tag, kind in tags:
        emb, ids = load_corpus_emb(tag)
        if kind == "ft":
            m = load_ft(a.name)
            qv = encode_q(m, vq, prefixes(a.name)[0])
        else:
            key = "bge-m3" if tag.startswith("bge") else tag.split("_s")[0]
            m = load_model(key, 256)
            qv = encode_queries(m, key, vq, with_filters=True)
        del m
        torch.cuda.empty_cache()
        for mode in ("all", "geo"):
            out = search(qv, emb, vq, item_locs, regions, k=300, mode=mode)
            row = {"модель": tag, "режим": mode}
            for k in (50, 100, 300):
                r = recall_at_k(to_preds(out, vq, ids, k), labels, k)
                row[f"R@{k}"] = r.mean()
                if k == 50:
                    per[(tag, mode)] = r
            res.append(row)
            print(row, flush=True)
        del emb
    df = pd.DataFrame(res)
    print(df.round(4).to_string(index=False))
    ft = f"{a.name}_s256"
    for r in a.ref:
        for mode in ("all", "geo"):
            d = boot(per[(ft, mode)], per[(r, mode)])
            print(f"{ft} − {r} ({mode}): {d[0]:+.4f} [{d[1]:+.4f}; {d[2]:+.4f}]")
    st = slice_table({f"{t}": per[(t, 'geo')] for t, _ in tags}, vq)
    print("срезы, R@50 geo:\n" + st.round(3).to_string())
    FT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{t}|{m}": s for (t, m), s in per.items()}).to_parquet(FT / f"dense_perq_{a.name}.parquet")
    df.to_csv(FT / f"dense_{a.name}.csv", index=False)
    st.to_csv(FT / f"dense_slices_{a.name}.csv")


# ------------------------------------------------------------------ пайплайн

def rule_for(locs, regions):
    """Гео-правило запроса по индексу (город / регион / вся Россия), как в итоговой гео-схеме geo2."""
    def f(i):
        l = locs[i]
        if l == RUSSIA:
            return GEO["russia"]
        if l in regions:
            return GEO["region"]
        return GEO["city"]
    return f


def run_multi(queries, rule_of, sp, dense_srcs, maps, weights=None):
    """Как geo_run.run (режим final), но с любым числом dense-источников.
    dense_srcs — список (D на GPU, эмбеддинги запросов). weights — веса RRF по источникам
    [sparse, dense1, ...] (None = все 1, как в geo.rrf). Возвращает {query_id: (позиции, скоры)}."""
    n = len(queries)
    locs = queries["search_location_id"].to_numpy()
    texts = queries["search_query"].to_numpy()
    qids = queries["query_id"].to_numpy()
    named = [maps.city.loc(t) for t in texts]
    kept = {}
    t0 = time.time()
    bs = 128
    for b0 in range(0, n, bs):
        dscs = []
        with torch.no_grad():
            for D, qe in dense_srcs:
                dscs.append((torch.from_numpy(qe[b0:b0 + bs]).to(D.device, D.dtype) @ D.T).float().cpu().numpy())
        for j in range(dscs[0].shape[0]):
            i = b0 + j
            ssc = sp.scores(texts[i])
            g_s = SparseIndex._top(np.arange(len(ssc)), ssc, 2 * K)
            rule = rule_of(i)
            tiers = maps.tiers(maps.base_rule(rule), int(locs[i]), named[i])
            ls, _ = tiered_top(ssc, tiers, K, g_s, positive=True)
            lists = [ls]
            for dsc in dscs:
                dv = dsc[j]
                g_d = np.argpartition(-dv, 2 * K)[:2 * K]
                g_d = g_d[np.argsort(-dv[g_d], kind="stable")]
                lists.append(tiered_top(dv, tiers, K, g_d, positive=False)[0])
            bmask, w = maps.bonus_mask(rule, int(locs[i]))
            if weights is None:
                kept[qids[i]] = rrf(lists, bonus=bmask, bonus_w=w)
            else:
                kept[qids[i]] = rrf_weighted(lists, weights, bonus=bmask, bonus_w=w)
        if (b0 // bs) % 5 == 0:
            print(f"  {min(b0 + bs, n)}/{n} за {time.time() - t0:.0f} с", flush=True)
    return kept


def rrf_weighted(lists, weights, k_rrf=60, n_out=300, bonus=None, bonus_w=0.0):
    """geo.rrf с весами источников: score = Σ w_s / (k_rrf + rank_s)."""
    cat = np.concatenate(lists)
    s = np.concatenate([w / (k_rrf + np.arange(1, len(l) + 1)) for l, w in zip(lists, weights)])
    u, first, inv = np.unique(cat, return_index=True, return_inverse=True)
    score = np.bincount(inv, weights=s)
    if bonus is not None and bonus_w:
        score = score + bonus_w * bonus[u]
    o = np.lexsort((first, -score))[:n_out]
    return u[o], score[o]


def variants(names):
    """Варианты источников: имя -> (список тегов dense, веса RRF или None)."""
    v = {"bge": (["bge-m3_s256"], None)}
    for nm in names:
        v[f"{nm}"] = ([f"{nm}_s256"], None)
        v[f"bge+{nm}"] = (["bge-m3_s256", f"{nm}_s256"], None)
        # три источника, у каждого dense вес 0.5: баланс sparse:dense и шкала скора как в двухисточниковой схеме
        v[f"bge+{nm}_w"] = (["bge-m3_s256", f"{nm}_s256"], [1.0, 0.5, 0.5])
    # пары дообученных моделей (например, bge-m3-lora + e5-small-ft) с тем же весом 0.5 у каждого dense
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            v[f"{n1}+{n2}_w"] = ([f"{n1}_s256", f"{n2}_s256"], [1.0, 0.5, 0.5])
    return v


def encode_for(tag, queries):
    """Эмбеддинги запросов для тега корпуса: bge-m3 — готовая модель, иначе дообученная из work/models."""
    if tag == "bge-m3_s256":
        m = load_model("bge-m3", 256)
        q = encode_queries(m, "bge-m3", queries, with_filters=True)
    else:
        nm = tag[:-len("_s256")]
        m = load_ft(nm)
        q = encode_q(m, queries, prefixes(nm)[0])
    del m
    torch.cuda.empty_cache()
    return q


def cmd_pipe(a):
    """Варианты источников (bge / ft / bge+ft) в пайплайне geo2 + бонусы метаданных, сравнение с 0.869 бутстрепом."""
    t0 = time.time()
    vq, labels, mask = load_validation()
    vq["query_id"] = vq["query_id"].astype(str)
    regions = region_location_ids()
    ids, corpus, train, sp, Dbge = prepare("lemma")
    maps = build_geo(corpus, train, train[mask], regions)
    print(f"подготовка {time.time() - t0:.0f} с", flush=True)
    V = variants(a.name)
    todo = a.variants or list(V)
    tags = sorted({t for v in todo for t in V[v][0]})
    D = {"bge-m3_s256": Dbge}
    for t in tags:
        if t not in D:
            e, i2 = load_corpus_emb(t)
            assert (i2 == ids).all()
            D[t] = torch.from_numpy(e).to("cuda", torch.float16)
    qv = {t: encode_for(t, vq) for t in tags}
    vlocs = vq["search_location_id"].to_numpy()
    FT.mkdir(parents=True, exist_ok=True)
    raw = {}
    for v in todo:
        srcs, w = V[v]
        t1 = time.time()
        kept = run_multi(vq, rule_for(vlocs, regions), sp, [(D[t], qv[t]) for t in srcs], maps, w)
        raw[v] = save_cands(kept, vq, ids, FT / f"raw_{v}_val.parquet")
        print(f"{v}: R@50 {recall_series(raw[v], labels).mean():.4f} ({time.time() - t1:.0f} с)", flush=True)
    del sp, train, corpus, D, Dbge
    torch.cuda.empty_cache()

    # бонусы метаданных: классификатор на train[mask], как в experiments/rerank_meta_geo2.py
    items = rm.load_items(DATA / "benchmark_items.parquet")
    tr = pd.read_parquet(DATA / "train.parquet", columns=["search_query", "search_infm_params_text", "item_microcat_id"])
    model = rm.fit_microcat(tr[mask])
    del tr
    ref = pd.read_parquet(CANDS / "hybrid_geo2_meta_val.parquet")
    ref_r = recall_series(ref, labels)
    qids = np.array(sorted(labels))
    perm = np.random.default_rng(0).permutation(len(qids))
    ib = ref_r.index.isin(set(qids[perm[len(qids) // 2:]]))
    rows, per = [], {"итог 0.869 (bge)": ref_r}
    for v in todo:
        f = features(raw[v], vq, items, model)
        c = rm.rerank_bonus(f, base="score", **W_META)
        del f
        c.to_parquet(FT / f"meta_{v}_val.parquet", index=False)
        r50, r300 = recall_series(c, labels), recall_series(c, labels, 300)
        per[v] = r50
        dA, dB = boot(r50, ref_r), boot(r50[ib], ref_r[ib])
        rows.append({"вариант": v, "geo2 R@50": recall_series(raw[v], labels).mean(), "+мета R@50": r50.mean(),
                     "R@300": r300.mean(), "Δ к 0.869": dA[0], "ДИ": f"[{dA[1]:+.4f}; {dA[2]:+.4f}]",
                     "B: R@50": r50[ib].mean(), "B: Δ": dB[0], "B: ДИ": f"[{dB[1]:+.4f}; {dB[2]:+.4f}]",
                     "A: R@50": r50[~ib].mean()})
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    print(f"база 0.869: все {ref_r.mean():.4f}, A {ref_r[~ib].mean():.4f}, B {ref_r[ib].mean():.4f}")
    print(tab.round(4).to_string(index=False))
    st = slice_table(per, vq)
    print("срезы (+мета, R@50):\n" + st.round(4).to_string())
    # срезы разницы с бутстрепом: без истории, текста нет в train, регион
    d = vq.set_index("query_id").loc[ref_r.index]
    sl = {"ответ без истории": ~d.target_has_history.values, "текста нет в train": ~d.text_in_train.values,
          "регион или вся Россия": d.is_region.values, "поиск по городу": ~d.is_region.values}
    srows = []
    for v in todo:
        for nm, mk in sl.items():
            b = boot(per[v][mk], ref_r[mk])
            srows.append({"вариант": v, "срез": nm, "n": int(mk.sum()), "база": ref_r[mk].mean(),
                          "R@50": per[v][mk].mean(), "Δ": b[0], "ДИ": f"[{b[1]:+.4f}; {b[2]:+.4f}]"})
    stab = pd.DataFrame(srows)
    print(stab.round(4).to_string(index=False))
    tab.to_csv(FT / f"pipe_{a.tag}.csv", index=False)
    st.to_csv(FT / f"pipe_slices_{a.tag}.csv")
    stab.to_csv(FT / f"pipe_slice_delta_{a.tag}.csv", index=False)
    pd.DataFrame(per).to_parquet(FT / f"pipe_perq_{a.tag}.parquet")
    print(f"всего {time.time() - t0:.0f} с")


def cmd_final(a):
    """Кандидаты лучшего варианта (a.best) -> work/cands/hybrid_ft_{val,bench}.parquet (после бонусов метаданных).
    Валидация: карты, классификатор и дообученная модель — по train[mask]; бенчмарк: карты и классификатор
    по полному train, дообученная модель — --bench-name (та же схема обучения на полном train, --full-train)."""
    t0 = time.time()
    v = variants(a.names)[a.best]
    srcs, w = v
    if a.bench_name:
        # для бенчмарка — та же модель, переобученная на полном train (как карты и классификатор)
        srcs = [f"{a.bench_name}_s256" if t == f"{a.name}_s256" else t for t in srcs]
    print(f"бенчмарк: источники {srcs}, веса {w}", flush=True)
    # валидация уже посчитана в cmd_pipe
    val = pd.read_parquet(FT / f"meta_{a.best}_val.parquet")
    val.to_parquet(CANDS / "hybrid_ft_val.parquet", index=False)
    regions = region_location_ids()
    ids, corpus, train, sp, Dbge = prepare("lemma")
    maps_b = build_geo(corpus, train, train, regions)
    del corpus, train
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    bq["query_id"] = bq["query_id"].astype(str)
    D = {"bge-m3_s256": Dbge}
    for t in srcs:
        if t not in D:
            e, i2 = load_corpus_emb(t)
            assert (i2 == ids).all()
            D[t] = torch.from_numpy(e).to("cuda", torch.float16)
    qv = {t: encode_for(t, bq) for t in srcs}
    kept = run_multi(bq, rule_for(bq["search_location_id"].to_numpy(), regions), sp,
                     [(D[t], qv[t]) for t in srcs], maps_b, w)
    cb = save_cands(kept, bq, ids, FT / f"raw_{a.best}_bench.parquet")
    del sp, D, Dbge
    items = rm.load_items(DATA / "benchmark_items.parquet")
    tr = pd.read_parquet(DATA / "train.parquet", columns=["search_query", "search_infm_params_text", "item_microcat_id"])
    model_b = rm.fit_microcat(tr)
    del tr
    nb = rm.rerank_bonus(features(cb, bq, items, model_b), base="score", **W_META)
    assert nb.groupby("query_id").size().eq(K).all() and nb.query_id.nunique() == len(bq)
    nb.to_parquet(CANDS / "hybrid_ft_bench.parquet", index=False)
    old = pd.read_parquet(CANDS / "hybrid_geo2_meta_bench.parquet")
    same = nb[nb["rank"] <= 50].merge(old[old["rank"] <= 50], on=["query_id", "item_id"]).shape[0] / (old["rank"] <= 50).sum()
    print(f"бенчмарк: {nb.query_id.nunique()} запросов; топ-50 совпадает с hybrid_geo2_meta_bench на {same:.3f}; "
          f"{time.time() - t0:.0f} с")


def main():
    """Разбор аргументов командной строки и запуск команды."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "dense", "pipe", "final"])
    ap.add_argument("--name", action="append", required=True, help="имя модели в work/models (можно несколько для pipe)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--best", default=None)
    ap.add_argument("--bench-name", default=None, help="final: модель для бенчмарка вместо --name (обученная на полном train)")
    a = ap.parse_args()
    a.names = list(a.name)
    if a.cmd in ("encode", "dense", "final"):
        a.name = a.name[0]
    {"encode": cmd_encode, "dense": cmd_dense, "pipe": cmd_pipe, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
