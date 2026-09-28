#!/bin/bash
# Очередь GPU-задач для eda/12_rerank.md (bge-reranker-v2-m3, скоры кешируются в work/rerank/)
PY=${PY:-python}   # интерпретатор окружения из requirements.txt
cd "$(dirname "$0")/.."
# 1) K=200 на 200 запросах подвыборки (оценка пользы большего K)
$PY experiments/rerank_ce_exp.py score bge-m3 200 geo2 200
# 2) бенчмарк: CE-скоры топ-100 geo2 (слияние выбирается позже по валидации)
$PY - <<'PY'
import sys, time; sys.path.insert(0, "src")
import pandas as pd, rerank_ce as R
q = pd.read_parquet("data/benchmark_queries.parquet")
c = pd.read_parquet("work/cands/hybrid_geo2_bench.parquet")
t = time.time()
R.ce_scores(c, q, "bge-m3", 100, cache=R.CACHE / "ce_bge-m3_bench.parquet")
print(f"bench K=100: {time.time() - t:.0f} с, {(time.time() - t) / c.query_id.nunique() * 1000:.0f} мс/запрос", flush=True)
PY
# 3) старые кандидаты geo на той же подвыборке 800 (для сравнения)
$PY experiments/rerank_ce_exp.py score bge-m3 100 geo 800
echo ALL_DONE
