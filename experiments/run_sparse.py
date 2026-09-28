"""Замеры sparse-ретривера на валидации и выгрузка кандидатов (запуск из корня репозитория).

python experiments/run_sparse.py            # все варианты текста × режимы, выбор лучшего, выгрузка топ-300
Результаты печатаются; кандидаты лучшего варианта -> work/cands/sparse_{val,bench}_{all,geo}.parquet
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sparse import DATA, WORK, SparseIndex, load_items, to_preds  # noqa: E402
from validation import load_validation, recall_at_k, slices  # noqa: E402

K = 300
VARIANTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["a", "b", "c"]


def main():
    items = load_items()
    val_q, val_labels, _ = load_validation()
    bench_q = pd.read_parquet(DATA / "benchmark_queries.parquet")
    res, timing, best = [], {}, (None, -1.0)
    for v in VARIANTS:
        idx = SparseIndex.build(items, v)
        # проверка: наш подсчёт скоров совпадает с bm25s.get_scores
        q0 = val_q["search_query"].iloc[0]
        from sparse import tokenize
        ref = idx.bm25.get_scores(list(dict.fromkeys(t for t in tokenize([q0])[0] if t in idx.vocab)))
        assert np.allclose(ref, idx.scores(q0), atol=1e-4), "скоры расходятся с bm25s"
        for mode in ("all", "geo"):
            c = idx.search(val_q, mode, K)
            timing[(v, mode)] = (idx.build_sec, idx.ms_per_query)
            r = recall_at_k(to_preds(c, 50), val_labels, 50)
            res.append({"вариант": v, "режим": mode, "recall@50": r.mean(),
                        "build, с": idx.build_sec, "мс/запрос": idx.ms_per_query})
            print(res[-1], flush=True)
            if mode == "geo" and r.mean() > best[1]:
                best = (v, r.mean())
        del idx
        gc.collect()
    print(pd.DataFrame(res).to_string())

    # лучший вариант (по geo): кривая, срезы, выгрузка
    v = best[0]
    print("\nЛучший вариант:", v)
    idx = SparseIndex.build(items, v)
    out = WORK / "cands"
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("all", "geo"):
        c = idx.search(val_q, mode, K)
        c.to_parquet(out / f"sparse_val_{mode}.parquet", index=False)
        curve = {f"@{k}": recall_at_k(to_preds(c, k), val_labels, k).mean() for k in (50, 100, 200, 300)}
        print(mode, "кривая", curve)
        r50 = recall_at_k(to_preds(c, 50), val_labels, 50)
        r50.to_frame().to_parquet(WORK / f"sparse_val_{mode}_perq.parquet")  # для разбора промахов
        print(slices(r50, val_q).round(4).to_string())
        t0 = time.time()
        cb = idx.search(bench_q, mode, K)
        cb.to_parquet(out / f"sparse_bench_{mode}.parquet", index=False)
        print(mode, "bench:", cb["query_id"].nunique(), "из", len(bench_q), "запросов,",
              len(cb), "строк,", f"{time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
