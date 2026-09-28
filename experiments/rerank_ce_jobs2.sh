#!/bin/bash
# Вторая очередь: после rerank_ce_jobs.sh пишет итоговые файлы (bge-reranker-v2-m3, K=100, RRF w=1).
PY=${PY:-python}   # интерпретатор окружения из requirements.txt
cd "$(dirname "$0")/.."
until grep -q ALL_DONE work/rerank/jobs.log; do sleep 10; done
# бенчмарк: скоры уже в кеше, считается только слияние
$PY experiments/rerank_ce.py --cands work/cands/hybrid_geo2_bench.parquet --queries bench --model bge-m3 --k 100 \
    --fusion rrf --out work/cands/hybrid_geo2_ce_bench.parquet
# валидация: досчитываются оставшиеся ~2200 запросов (~40 мин на общем GPU)
$PY experiments/rerank_ce.py --cands work/cands/hybrid_geo2_val.parquet --queries val --model bge-m3 --k 100 \
    --fusion rrf --out work/cands/hybrid_geo2_ce_val.parquet
echo ALL_DONE2
