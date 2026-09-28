"""Второй эмбеддер Qwen/Qwen3-Embedding-0.6B и ансамбли dense-моделей через RRF (см. eda/13_qwen_ensemble.md).

Модель (Apache-2.0, 0.6B, dim 1024, last-token pooling, токенизатор сам дописывает <|endoftext|>) берётся как есть,
без дообучения. Промпты — по карточке модели: у запроса инструкция «Instruct: {task}\\nQuery:{query}»,
у документа префикса нет.

Команды (из корня репозитория):
    python experiments/dense_qwen.py encode --max-seq 192     # корпус → work/emb/qwen3-0.6b_s192.{npy,_ids.npy,json}
    python experiments/dense_qwen.py run                      # замеры Qwen3 и ансамблей (старая гео-схема), кандидаты в work/cands:
                                                      #   qwen_{val,bench}_{all,geo}, hybrid3_{val,bench}_geo (BM25 + bge-m3 + Qwen3)
    CUDA_VISIBLE_DEVICES= python experiments/dense_qwen.py e5ctrl   # контроль: e5-small третьим источником (CPU)
    python experiments/qwen_geo2.py                           # Qwen3 третьим источником в гео-схеме geo2 + бонусы метаданных

Готовые модули (dense, text, validation, hybrid) только импортируются, не меняются.
Пик видеопамяти ограничен долей set_per_process_memory_fraction (GPU делится с другим процессом).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dense import DATA, EMB, ROOT, load_corpus_emb, query_text, search, to_frame, to_preds  # noqa: E402

NAME = "Qwen/Qwen3-Embedding-0.6B"
# Задача для инструкции запроса (по-английски, как в примерах карточки).
TASK = ("Given a search query from a Russian classifieds site for services, "
        "retrieve service listings that match the query")
# Вариант из config_sentence_transformers.json модели — для сравнения.
TASK_DEFAULT = "Given a web search query, retrieve relevant passages that answer the query"
CANDS = ROOT / "work" / "cands"
K = 300
MEM_FRACTION = 0.19  # ≈1.45 ГБ из 8 ГБ RTX 3070 Laptop (кодирование корпуса шло с 0.42; для запросов хватает весов fp16 ≈1.2 ГБ)


def q_prompt(task: str) -> str:
    """Префикс запроса по карточке: 'Instruct: {task}\\nQuery:' + текст запроса (без пробела после двоеточия)."""
    return f"Instruct: {task}\nQuery:"


def load_model(max_seq: int):
    """Qwen3-Embedding через sentence-transformers, fp16 на GPU; модули (last-token pooling + нормировка) — из репозитория."""
    from sentence_transformers import SentenceTransformer
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    m = SentenceTransformer(NAME, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = max_seq
    return m


def encode_corpus(max_seq: int, batch: int, chunk: int = 8192) -> None:
    """Кодирует item_text корпуса кусками (с продолжением после обрыва) и пишет npy/ids/json в work/emb."""
    from text import item_text
    tag = f"qwen3-0.6b_s{max_seq}"
    items = pd.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    docs = item_text(items).tolist()  # документ без префикса (карточка модели)
    ids = items["item_id"].astype(str).to_numpy()
    parts = EMB / f"{tag}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    model = load_model(max_seq)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_chunks = (len(docs) + chunk - 1) // chunk
    for c in range(n_chunks):
        f = parts / f"{c:04d}.npy"
        if f.exists():
            continue
        e = model.encode(docs[c * chunk:(c + 1) * chunk], batch_size=batch, prompt="",
                         normalize_embeddings=True, convert_to_numpy=True)
        np.save(f, e.astype(np.float16))
        print(f"[{tag}] {min((c + 1) * chunk, len(docs))}/{len(docs)} за {time.time() - t0:.0f} с", flush=True)
    secs = time.time() - t0
    emb = np.concatenate([np.load(parts / f"{c:04d}.npy") for c in range(n_chunks)])
    np.save(EMB / f"{tag}.npy", emb)
    np.save(EMB / f"{tag}_ids.npy", ids)
    meta = dict(model=NAME, license="apache-2.0", max_seq=max_seq, batch=batch, precision="fp16",
                gpu=torch.cuda.get_device_name(0), pooling="lasttoken (+<|endoftext|>), normalize",
                q_prompt=q_prompt(TASK), d_prompt="", n=len(ids), dim=int(emb.shape[1]),
                seconds_this_run=round(secs, 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    (EMB / f"{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(meta, flush=True)


def encode_queries(model, queries: pd.DataFrame, task: str, with_filters: bool = True) -> np.ndarray:
    """Эмбеддинги запросов: инструкция + текст запроса (+ значения фильтров, как у bge-m3)."""
    texts = [q_prompt(task) + t for t in query_text(queries, with_filters)]
    return model.encode(texts, batch_size=8, normalize_embeddings=True, prompt="",
                        convert_to_numpy=True).astype(np.float32)


# ---------------------------------------------------------------- ансамбли

def load_src(path: Path, queries: pd.DataFrame, item_loc: pd.Series) -> pd.DataFrame:
    """Кандидаты источника в формате hybrid.load_cands: query_id, item_id (str), score, rank, city."""
    qloc = queries.set_index("query_id")["search_location_id"]
    c = pd.read_parquet(path)
    c["query_id"] = c["query_id"].astype(str)
    c["item_id"] = c["item_id"].astype(str)
    c["city"] = c["item_id"].map(item_loc).to_numpy() == c["query_id"].map(qloc).to_numpy()
    return c.sort_values(["query_id", "rank"], kind="stable").reset_index(drop=True)


def ensemble(srcs_geo: dict, srcs_all: dict, names: list, query_ids) -> pd.DataFrame:
    """RRF k=60 без ярусов по geo-спискам источников names, добор — all-списками тех же источников (как в hybrid.py)."""
    from hybrid import fuse_rrf, top_n
    fused = fuse_rrf({n: srcs_geo[n] for n in names}, 60, tiers=False)
    return top_n(fused, query_ids, [srcs_all[n] for n in names])


def pairs_top(df: pd.DataFrame, k: int) -> set:
    d = df[df["rank"] <= k]
    return set(zip(d["query_id"], d["item_id"]))


def run(tag: str, max_seq: int) -> None:
    from hybrid import bootstrap_diff, item_locations, recall_curve
    from hybrid import to_preds as df_preds
    from validation import load_validation, recall_at_k, region_location_ids, slices

    emb, ids = load_corpus_emb(tag)
    items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_location_id"])
    assert (items["item_id"].astype(str).to_numpy() == ids).all(), "порядок item_id не совпадает с эмбеддингами"
    item_locs = items["item_location_id"].to_numpy()
    vq, vl, _ = load_validation()
    val_regions = set(vq.loc[vq["is_region"], "search_location_id"])
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    meta = json.loads((EMB / f"{tag}.json").read_text())
    prompt = meta["q_prompt"]  # промпт запроса ровно тот, с которым согласован корпус (d_prompt = "")
    assert prompt == q_prompt(TASK)

    # --- 1. Эмбеддинги запросов (валидация и бенчмарк) с кешем в work/emb; модель выгружается до поиска,
    # чтобы на GPU одновременно не лежали веса (≈1.2 ГБ) и матрица корпуса (≈0.39 ГБ)
    qfile = {s: EMB / f"{tag}_q_{s}.npy" for s in ("val", "bench")}
    if not all(f.exists() for f in qfile.values()):
        model = load_model(max_seq)
        torch.cuda.reset_peak_memory_stats()
        for s, q in (("val", vq), ("bench", bq)):
            t = time.time()
            e = encode_queries(model, q, TASK)
            np.save(qfile[s], e)
            print(f"запросы {s}: {len(q)} за {time.time() - t:.1f} с ({(time.time() - t) / len(q) * 1000:.2f} мс/запрос), "
                  f"пик VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} ГБ", flush=True)
        del model
        import gc
        gc.collect()  # у SentenceTransformer есть циклические ссылки — без gc веса остаются на GPU
        torch.cuda.empty_cache()
    qe, b_emb = np.load(qfile["val"]), np.load(qfile["bench"])

    res = {}
    t = time.time()
    for mode in ("all", "geo"):
        res[mode] = search(qe, emb, vq, item_locs, val_regions, k=K, mode=mode)
    print(f"поиск all+geo: {(time.time() - t) / len(vq) * 1000:.2f} мс/запрос; R@50 all "
          f"{recall_at_k(to_preds(res['all'], vq, ids, 50), vl).mean():.4f}, geo "
          f"{recall_at_k(to_preds(res['geo'], vq, ids, 50), vl).mean():.4f}", flush=True)

    CANDS.mkdir(exist_ok=True)
    for mode in ("all", "geo"):
        to_frame(res[mode], vq, ids).to_parquet(CANDS / f"qwen_val_{mode}.parquet", index=False)
    b_regions = region_location_ids()
    for mode in ("all", "geo"):
        df = to_frame(search(b_emb, emb, bq, item_locs, b_regions, k=K, mode=mode), bq, ids)
        assert df.groupby("query_id").size().eq(K).all() and df["query_id"].nunique() == len(bq)
        df.to_parquet(CANDS / f"qwen_bench_{mode}.parquet", index=False)
    torch.cuda.empty_cache()

    iloc = item_locations()
    vq["query_id"] = vq["query_id"].astype(str)
    bq["query_id"] = bq["query_id"].astype(str)
    files = {"bm25": "sparse", "bge": "dense", "qwen": "qwen"}
    sv = {m: {n: load_src(CANDS / f"{f}_val_{m}.parquet", vq, iloc) for n, f in files.items()} for m in ("all", "geo")}

    # одиночные источники: кривые, срезы, бутстреп Qwen − bge-m3
    per_q = {}
    print("\nОдиночные источники, Recall@k")
    for n in ("bm25", "bge", "qwen"):
        for m in ("all", "geo"):
            cur = recall_curve(sv[m][n], vl)
            print(f"  {n:5s} {m}: " + ", ".join(f"@{k} {v:.4f}" for k, v in cur.items()))
        per_q[n] = recall_at_k(df_preds(sv["geo"][n], 50), vl)
    for m in ("all", "geo"):
        a = recall_at_k(df_preds(sv[m]["qwen"], 50), vl)
        b = recall_at_k(df_preds(sv[m]["bge"], 50), vl)
        d, lo, hi = bootstrap_diff(a, b)
        print(f"  бутстреп qwen − bge-m3 ({m}) @50: {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")
        a3 = recall_at_k(df_preds(sv[m]["qwen"], 300), vl, 300)
        b3 = recall_at_k(df_preds(sv[m]["bge"], 300), vl, 300)
        d, lo, hi = bootstrap_diff(a3, b3)
        print(f"  бутстреп qwen − bge-m3 ({m}) @300: {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")

    # --- 2. Ансамбли
    combos = {"BM25 + bge-m3 (контроль)": ["bm25", "bge"], "BM25 + Qwen3": ["bm25", "qwen"],
              "BM25 + bge-m3 + Qwen3": ["bm25", "bge", "qwen"], "bge-m3 + Qwen3": ["bge", "qwen"]}
    hyb = {c: ensemble(sv["geo"], sv["all"], ns, vq["query_id"]) for c, ns in combos.items()}
    ctrl = "BM25 + bge-m3 (контроль)"
    for c, h in hyb.items():
        per_q[c] = recall_at_k(df_preds(h, 50), vl)
    print("\nАнсамбли (RRF k=60, geo), Recall@50 / @300 и Δ к контролю")
    for c, h in hyb.items():
        cur = recall_curve(h, vl)
        d, lo, hi = bootstrap_diff(per_q[c], per_q[ctrl])
        print(f"  {c:28s} @50 {cur[50]:.4f} @100 {cur[100]:.4f} @200 {cur[200]:.4f} @300 {cur[300]:.4f} "
              f"| Δ@50 {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")
        r3a = recall_at_k(df_preds(h, 300), vl, 300)
        r3b = recall_at_k(df_preds(hyb[ctrl], 300), vl, 300)
        d, lo, hi = bootstrap_diff(r3a, r3b)
        print(f"  {'':28s} Δ@300 {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")

    best = max((c for c in combos if c != ctrl), key=lambda c: per_q[c].mean())
    print(f"\nлучший ансамбль: {best}")
    # устойчивость выбора: две половины валидации
    rng = np.random.default_rng(0)
    half = set(rng.choice(vq["query_id"].to_numpy(), len(vq) // 2, replace=False))
    for part in ("A", "B"):
        msk = {c: per_q[c][per_q[c].index.isin(half) == (part == "A")].mean() for c in combos}
        print(f"  половина {part}: " + ", ".join(f"{c} {v:.4f}" for c, v in msk.items()))

    sla = pd.concat({f"{n} all": slices(recall_at_k(df_preds(sv["all"][n], 50), vl), vq)["recall"]
                     for n in ["bge", "qwen"]}, axis=1)
    print("\nСрезы Recall@50 all\n", sla.round(3).to_string(), sep="")
    sl = pd.concat({n: slices(per_q[n], vq)["recall"] for n in ["bm25", "bge", "qwen", ctrl, best]}, axis=1)
    sl.insert(0, "запросов", slices(per_q[ctrl], vq)["запросов"].astype(int))
    print("\nСрезы Recall@50 geo\n", sl.round(3).to_string(), sep="")

    # сколько правильных пар добавил третий источник
    gold = {(q, i) for q, rel in vl.items() for i in rel}
    h2, h3 = pairs_top(hyb[ctrl], 50) & gold, pairs_top(hyb[best], 50) & gold
    s50 = {n: pairs_top(sv["geo"][n], 50) & gold for n in ("bm25", "bge", "qwen")}
    s300 = {n: pairs_top(sv["geo"][n], 300) & gold for n in ("bm25", "bge", "qwen")}
    only_q50 = s50["qwen"] - s50["bm25"] - s50["bge"]
    print(f"\nпар (запрос, правильный id): {len(gold)}")
    print(f"  контроль@50 нашёл {len(h2)}, лучший@50 {len(h3)}: +{len(h3 - h2)} новых, −{len(h2 - h3)} потерянных")
    print(f"  Qwen3 топ-50 нашёл пар, которых нет в топ-50 BM25 и bge-m3: {len(only_q50)}; "
          f"из них в лучшем@50: {len(only_q50 & h3)}")
    print(f"  то же в топ-300: {len(s300['qwen'] - s300['bm25'] - s300['bge'])}")
    # потолок: recall объединения топ-300 источников по запросам
    for n_src, names in (("2", ["bm25", "bge"]), ("3", ["bm25", "bge", "qwen"])):
        u = pd.concat([sv["geo"][n][["query_id", "item_id"]] for n in names]).groupby("query_id")["item_id"].agg(set)
        print(f"  Recall объединения топ-300 ({n_src} источника): "
              f"{np.mean([len(rel & u.get(q, set())) / len(rel) for q, rel in vl.items()]):.4f}")

    # --- 3. Кандидаты лучшего ансамбля (val и bench)
    names = combos[best]
    hyb[best].to_parquet(CANDS / "hybrid3_val_geo.parquet", index=False)
    sb = {m: {n: load_src(CANDS / f"{files[n]}_bench_{m}.parquet", bq, iloc) for n in names} for m in ("all", "geo")}
    hb = ensemble(sb["geo"], sb["all"], names, bq["query_id"])
    n_per = hb.groupby("query_id").size().reindex(bq["query_id"]).fillna(0)
    assert (n_per == K).all(), "у части запросов бенчмарка не 300 кандидатов"
    hb.to_parquet(CANDS / "hybrid3_bench_geo.parquet", index=False)
    old = load_src(CANDS / "hybrid_bench_geo.parquet", bq, iloc)
    ov = len(pairs_top(hb, 50) & pairs_top(old, 50)) / (50 * len(bq))
    print(f"\nбенчмарк: {len(bq)} запросов; совпадение топ-50 с hybrid_bench_geo (BM25 + bge-m3): {ov:.1%}")
    print("записано: work/cands/qwen_{val,bench}_{all,geo}.parquet, work/cands/hybrid3_{val,bench}_geo.parquet")


def e5_control() -> None:
    """Контроль цены: третий источник — e5-small seq 256 (корпус на GPU за 145 с, work/emb/e5-small_s256) вместо Qwen3.
    Запросы и поиск — на CPU (GPU занят), кандидаты → work/cands/e5small_val_{all,geo}.parquet."""
    from dense import encode_queries as enc, load_model as load_st
    from hybrid import bootstrap_diff, item_locations, recall_curve
    from hybrid import to_preds as df_preds
    from validation import load_validation, recall_at_k
    vq, vl, _ = load_validation()
    emb, ids = load_corpus_emb("e5-small_s256")
    items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_location_id"])
    assert (items["item_id"].astype(str).to_numpy() == ids).all()
    m = load_st("e5-small", 256, quantize=False)
    qe = enc(m, "e5-small", vq, with_filters=True)
    regs = set(vq.loc[vq["is_region"], "search_location_id"])
    for mode in ("all", "geo"):
        r = search(qe, emb.astype(np.float32), vq, items["item_location_id"].to_numpy(), regs, K, mode)
        to_frame(r, vq, ids).to_parquet(CANDS / f"e5small_val_{mode}.parquet", index=False)
    vq["query_id"] = vq["query_id"].astype(str)
    iloc = item_locations()
    files = {"bm25": "sparse", "bge": "dense", "qwen": "qwen", "e5": "e5small"}
    sv = {mo: {n: load_src(CANDS / f"{f}_val_{mo}.parquet", vq, iloc) for n, f in files.items()} for mo in ("all", "geo")}
    print(f"e5-small s256 geo R@50 {recall_at_k(df_preds(sv['geo']['e5'], 50), vl).mean():.4f}")
    pq = {}
    for name, ns in {"BM25 + bge-m3": ["bm25", "bge"], "+ e5-small": ["bm25", "bge", "e5"],
                     "+ Qwen3": ["bm25", "bge", "qwen"], "+ Qwen3 + e5-small": ["bm25", "bge", "qwen", "e5"]}.items():
        h = ensemble(sv["geo"], sv["all"], ns, vq["query_id"])
        pq[name] = recall_at_k(df_preds(h, 50), vl)
        cur = recall_curve(h, vl)
        d, lo, hi = bootstrap_diff(pq[name], pq["BM25 + bge-m3"])
        print(f"  {name:20s} @50 {cur[50]:.4f} @300 {cur[300]:.4f} | Δ@50 {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")
    d, lo, hi = bootstrap_diff(pq["+ Qwen3"], pq["+ e5-small"])
    print(f"  Qwen3 − e5-small третьим источником: {d:+.4f} [{lo:+.4f}; {hi:+.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "run", "e5ctrl"])
    ap.add_argument("--max-seq", type=int, default=192)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "encode":
        encode_corpus(a.max_seq, a.batch)
    elif a.cmd == "e5ctrl":
        torch.set_num_threads(4)
        e5_control()
    else:
        run(f"qwen3-0.6b_s{a.max_seq}", a.max_seq)
