# Эксперименты

Скрипты экспериментов, которые не входят в итоговое решение. Их результаты описаны в отчётах `eda/`, и на эти скрипты
ссылаются отчёты. Итоговый путь (`src/pipeline.py` и обучение из `train_all.sh`) от них не зависит.

Скрипты запускаются из корня репозитория, например `python experiments/hybrid.py`, и импортируют модули из `src/`.
Многим нужны промежуточные файлы в `work/` от предыдущих шагов. Какие именно, написано в docstring каждого скрипта.

| скрипт | что проверял | отчёт |
|---|---|---|
| `check_validation.py` | похожа ли валидация на бенчмарк, нет ли утечки, простые бейзлайны, разброс по seed | `eda/05_validation.md` |
| `run_sparse.py` | BM25 со стеммингом: замеры и кандидаты | `eda/06_sparse.md` |
| `dense_compare.py`, `dense_run.py` | сравнение эмбеддеров (e5, USER, bge-m3), кандидаты dense | `eda/07_dense.md` |
| `rank_curve.py` | кривая Recall@K до 1000 и место правильного ответа во всём корпусе | `eda/07_dense.md` |
| `hybrid.py` | слияние BM25 и эмбеддингов через RRF | `eda/08_hybrid.md` |
| `run_sparse_lemma.py`, `lemma_report.py` | стемы, леммы и их сочетание для BM25 | `eda/09_lemma.md` |
| `geo_report.py`, `geo_final_report.py` | таблицы по вариантам гео-правил и итоговой гео-схеме | `eda/10_geo.md` |
| `rerank_meta_geo2.py` | бонусы метаданных поверх кандидатов гео-схемы (сабмит 1) | `eda/11_meta_boost.md` |
| `rerank_ce.py`, `rerank_ce_exp.py`, `rerank_ce_report.py`, `rerank_ce_jobs*.sh` | cross-encoder bge-reranker-v2-m3 поверх кандидатов | `eda/12_rerank.md` |
| `dense_qwen.py`, `qwen_geo2.py` | Qwen3-Embedding-0.6B отдельно и третьим источником в RRF | `eda/13_qwen_ensemble.md` |
| `error_analysis.py` | разбор промахов схемы сабмита 1 | `eda/14_errors.md` |
| `rerank_quality_exp.py` | бонусы за качество объявления | `eda/15_quality.md` |
