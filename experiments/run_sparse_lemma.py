"""Сравнение stem / lemma / lemma+stem для BM25 (запуск из корня репозитория).

python experiments/run_sparse_lemma.py              # все три схемы, текст документа b (заголовок ×2 + item_text)
python experiments/run_sparse_lemma.py lemma        # только выбранные схемы (через запятую)

Для каждой схемы строится индекс, ищутся топ-300 на валидации и бенчмарке (all, geo).
Промежуточное -> work/lemma/: кандидаты {scheme}_{val,bench}_{mode}.parquet,
recall по запросам perq.parquet, время timing.csv. Выгрузку лучшей схемы в work/cands
делает experiments/lemma_report.py после сравнения (сами sparse_*.parquet не трогаем).
"""
from __future__ import annotations

import gc
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sparse import DATA, WORK, load_items, to_preds  # noqa: E402
from sparse_lemma import SCHEMES, LemmaIndex  # noqa: E402
from validation import load_validation, recall_at_k  # noqa: E402

K = 300
OUT = WORK / "lemma"


def main():
    schemes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SCHEMES)
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    val_q, val_labels, _ = load_validation()
    bench_q = pd.read_parquet(DATA / "benchmark_queries.parquet")
    perq, timing = [], []
    for sch in schemes:
        idx = LemmaIndex.build(items, scheme=sch, variant="b")
        # проверка: наш подсчёт скоров совпадает с bm25s.get_scores
        q0 = val_q["search_query"].iloc[0]
        ref = idx.bm25.get_scores(idx.query_cols(q0))
        assert np.allclose(ref, idx.scores(q0), atol=1e-4), "скоры расходятся с bm25s"
        row = {"scheme": sch, "build_sec": idx.build_sec, "lemma_sec": idx.mapper.lemma_sec,
               "words": idx.n_words, "terms": sum(1 for t in idx.vocab if t)}
        for mode in ("all", "geo"):
            c = idx.search(val_q, mode, K)
            row[f"ms_{mode}"] = idx.ms_per_query
            c.to_parquet(OUT / f"{sch}_val_{mode}.parquet", index=False)
            for k in (50, 100, 200, 300):
                r = recall_at_k(to_preds(c, k), val_labels, k)
                perq.append(pd.DataFrame({"query_id": r.index, "scheme": sch, "mode": mode, "k": k,
                                          "recall": r.to_numpy()}))
                if k == 50:
                    row[f"r50_{mode}"] = r.mean()
            cb = idx.search(bench_q, mode, K)
            cb.to_parquet(OUT / f"{sch}_bench_{mode}.parquet", index=False)
            row[f"bench_{mode}_q"] = cb["query_id"].nunique()
        row["maxrss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        timing.append(row)
        print(row, flush=True)
        del idx
        gc.collect()
    pd.concat(perq).to_parquet(OUT / ("perq.parquet" if len(schemes) == 3 else f"perq_{'_'.join(schemes)}.parquet"),
                               index=False)
    pd.DataFrame(timing).to_csv(OUT / ("timing.csv" if len(schemes) == 3 else f"timing_{'_'.join(schemes)}.csv"),
                                index=False)
    print(pd.DataFrame(timing).round(4).to_string())


if __name__ == "__main__":
    main()
