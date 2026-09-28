"""Сквозной воспроизводимый пайплайн решения: запрос -> 50 item_id из benchmark_items.

Запуск из корня репозитория (окружение из requirements.txt):

    python src/pipeline.py --mode val                          # всё строится на train[train_mask],
                                                               # печатает Recall@50 и срезы на валидации
    python src/pipeline.py --mode bench --out answer.csv       # всё строится на полном train,
                                                               # пишет answer.csv и проверяет формат
    --scheme meta                                              # прежняя схема без обучаемых моделей
                                                               # (сабмит 1: вал 0.869, платформа 0.8640)

Схема (подробности и цифры — SOLUTION.md и eda/*.md). Это RAG без генерации: офлайн индексируем корпус,
онлайн по запросу достаём кандидатов из трёх индексов, сливаем и переупорядочиваем обученным ранжировщиком.

    0. данные: корпус, train, запросы режима (val — отложенная валидация, bench — benchmark_queries);
    офлайн (один раз на корпус, кэшируется в work/); документ = заголовок + очищенные параметры
    + 500 символов описания (src/text.py):
      1. BM25 по леммам + стемам (src/sparse_lemma.py)                       -> work/pipeline/bm25_*.pkl
      2. эмбеддинги корпуса bge-m3 (src/dense.py, fp16, 256 токенов)         -> work/emb/bge-m3_s256.npy
         и дообученной bge-m3-lora-all (src/ft_train.py, src/ft_eval.py encode)  -> work/emb/bge-m3-lora-all_s256.npy
    обучаемые артефакты (строит train_all.sh, см. README.md «Как воспроизвести»):
         work/models/bge-m3-lora-all — LoRA-дообучение bge-m3 на всех парах train[train_mask] (запрос -> выбранное, до 20 на текст);
         work/ltr_ft_all/models/ltr_{0..4}.txt — 5 бустеров LightGBM lambdarank (src/ltr_big.py + src/ltr_ft.py);
    онлайн (на каждый запрос) и то, что учится на train (val — train[train_mask], bench — полный train):
      3. эмбеддинги запроса (+ значения фильтров) bge-m3 и bge-m3-lora-all;
      4. гео-карты: регион -> города, куда из него ходят; город -> соседи и частые пары (src/geo.py);
      5. топ-300 BM25 и топ-300 bge-m3 внутри гео-пула, RRF k=60 + гео-бонус (= geo2, сабмит 1 без бонусов),
         плюс топ-100 bge-m3-lora-all в том же пуле (новые кандидаты, ~18 на запрос) (src/ltr_ft.py);
      6. признаки пар запрос–кандидат: скоры и ранги источников, гео, микрокатегория (классификатор по train),
         «Вид/Тип услуги», качество объявления, косинус дообученной модели (src/ltr.py, ltr_big.py, ltr_ft.py);
      7. ранжирование: среднее z-оценок 5 бустеров -> топ-300;
      8. первые 50 -> answer.csv + проверка формата (check_answer.py).
    Схема meta вместо 5–7: только geo2 и ручные бонусы метаданных (src/rerank_meta.py).

Все существующие модули только импортируются: здесь нет своих копий BM25, гео-правил, признаков или моделей,
только порядок шагов, параметры выбранной конфигурации, кэш и проверки.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import random
import resource
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import rerank_meta as rm  # noqa: E402
from baselines import write_answer  # noqa: E402
from dense import MODELS, encode_corpus, encode_queries, load_corpus_emb, load_model  # noqa: E402
from geo import RUSSIA, GeoMaps, centroids, rrf, tiered_top  # noqa: E402
from sparse import SparseIndex  # noqa: E402
from sparse_lemma import LemmaIndex  # noqa: E402
from validation import add_query_key, load_validation, recall_at_k, region_location_ids, slices  # noqa: E402

ROOT = SRC.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
EMB = WORK / "emb"
OUT = WORK / "pipeline"          # кэш BM25, кандидаты, замеры времени этого пайплайна

# Файлы прежних экспериментов, с которыми сверяется пайплайн (если они есть): так видно,
# что чистая сборка воспроизводит проверенное решение (валидация 0.869, платформа 0.864).
REFERENCE = {
    ("val", "hybrid"): WORK / "cands" / "hybrid_geo2_val.parquet",        # после шага 7 (eda/10_geo.md)
    ("val", "meta"): WORK / "cands" / "hybrid_geo2_meta_val.parquet",     # после шага 8 (eda/11_meta_boost.md)
    ("val", "quality"): WORK / "cands" / "hybrid_geo2_meta_q_val.parquet",  # шаг 9 (eda/15_quality.md)
    ("bench", "hybrid"): WORK / "cands" / "hybrid_geo2_bench.parquet",
    ("bench", "meta"): ROOT / "submission_1_meta.csv",                    # сабмит 1, платформа 0.8640
    ("bench", "quality"): WORK / "cands" / "hybrid_geo2_meta_q_bench.parquet",
    ("val", "ltr"): WORK / "ltr_ft_all" / "hybrid_ltrft_val.parquet",      # src/ltr_ft.py final --out work/ltr_ft_all (вал 0.9156)
    ("bench", "ltr"): ROOT / "submission_5_ltrft_all.csv",                 # сабмит 5, платформа 0.9041
}


# ================================================================== конфигурация

@dataclass
class Config:
    """Параметры выбранной конфигурации. Каждый выбран на валидации, ссылка на отчёт — в комментарии."""
    seed: int = 42
    n_cands: int = 300          # топ-300 каждого источника: @300 = 0.942 — потолок для переранжирования
    n_answer: int = 50          # в ответе ровно 50 id (Recall@50, порядок внутри 50 не важен)
    # --- sparse (eda/06_sparse.md, eda/09_lemma.md): документ «заголовок ×2 + item_text» (вариант b),
    #     каждое слово -> два термина L:<лемма> и S:<стем>; BM25 lucene k1=1.5, b=0.75
    sparse_scheme: str = "lemma+stem"
    sparse_variant: str = "b"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # --- dense (eda/07_dense.md): BAAI/bge-m3, CLS-вектор, fp16 на GPU, 256 токенов документа;
    #     к запросу дописываются значения фильтров: «ремонт (Ремонт и отделка)» (+0.01–0.02 R@50)
    dense_model: str = "bge-m3"
    dense_max_seq: int = 256
    dense_tag: str = "bge-m3_s256"      # имя кэша эмбеддингов корпуса в work/emb/
    dense_query_filters: bool = True
    dense_batch: int = 128              # запросы кодируются и скорятся батчами по 128
    # --- слияние и гео (eda/08_hybrid.md, eda/10_geo.md)
    rrf_k: int = 60
    geo_city_rule: str = "SOFT_0.015"   # пул «свой город + соседи ≤100 км + пары ≥1%», +0.015 своему городу
    geo_region_rule: str = "RSOFT_0.02"  # пул «все города региона из train», +0.02 городам с долей ≥1%
    geo_russia_rule: str = "RSOFT_0.005"  # «вся Россия» (621540): тот же пул, бонус слабее
    # --- бонусы метаданных (eda/11_meta_boost.md): выбраны на половине A валидации, база = score (с гео-бонусом)
    meta_weights: dict = field(default_factory=lambda: dict(w_vid=0.003, w_tip=0.0015, w_mc=0.007, w_prf=0.002))
    meta_prf_top: int = 20
    # --- ОПЦИОНАЛЬНО, по умолчанию выключено: мягкий бонус качества (eda/15_quality.md).
    #     На валидации +0.007 [+0.004; +0.011], но на срезе «ответ без истории» (≈ бенчмарк) всего
    #     +0.004 [−0.003; +0.013]: отзывы во многом работают как популярность. Ожидание на платформе
    #     ≈ +0.004, в шуме. Включать отдельным сабмитом, чтобы увидеть эффект изолированно.
    quality: bool = False
    quality_weights: dict = field(default_factory=lambda: {"pct_rev_log": 0.0025, "pct_desc_log": 0.0019})
    # --- ИТОГОВАЯ схема (scheme="ltr", по умолчанию): дообученный эмбеддер + обученный ранжировщик.
    #     Вал 0.869 -> 0.9156, платформа 0.8640 -> 0.9041 (eda/17_finetune.md, 18_ltr_big.md, 19_ltr_ft.md)
    ltr: bool = True                    # False = схема meta (geo2 + ручные бонусы, сабмит 1)
    ft_model: str = "bge-m3-lora-all"   # work/models/<имя>: LoRA r=16 на всех 219k пар train[train_mask] (cap 20)
    ft_tag: str = "bge-m3-lora-all_s256"  # кэш эмбеддингов корпуса дообученной моделью в work/emb/
    ltr_models: str = "work/ltr_ft_all/models"  # 5 бустеров LightGBM + ltr.json (признаки, K_FT, N_FUSED)
    # --- не включено (проверено, прироста нет или он в шуме):
    extra_dense: str | None = None      # Qwen3-Embedding-0.6B третьим списком в RRF — не включать (eda/13)
    rerank_ce: bool = False             # cross-encoder bge-reranker-v2-m3 по топ-100: +0.008 в шуме (eda/12)
    max_vram_gb: float = 2.0            # потолок видеопамяти процесса (GPU общий)


# ================================================================== служебное: сид, замеры, кэш

def set_seed(seed: int) -> None:
    """Фиксирует генераторы. Сами шаги детерминированы (SGD классификатора с random_state=0,
    валидация отобрана заранее с seed 42), сид — страховка для библиотек."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


TIMINGS: list[dict] = []


@contextmanager
def step(name: str):
    """Замер шага: время, пик RAM процесса к концу шага, пик видеопамяти внутри шага."""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    print(f"\n=== {name}", flush=True)
    yield
    sec = time.time() - t0
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2   # КБ -> ГБ (Linux)
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if cuda else 0.0
    TIMINGS.append({"шаг": name, "сек": round(sec, 1), "пик RAM процесса, ГБ": round(ram, 2),
                    "пик VRAM шага, ГБ": round(vram, 2)})
    print(f"    {sec:.1f} с, RAM {ram:.1f} ГБ, VRAM {vram:.2f} ГБ", flush=True)


def fingerprint(*parts) -> str:
    """Короткий хэш параметров и входов: если он поменялся, кэш считается устаревшим."""
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:16]


def code_hash(*files: str) -> str:
    """Хэш исходников, от которых зависит индекс (текст документа, токенизация)."""
    return fingerprint(*[(SRC / f).read_bytes() for f in files])


# ================================================================== шаг 0: данные

ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text", "item_location_id",
             "item_latitude", "item_longitude", "item_microcat_id",
             # признаки качества (нужны только опциональному шагу 9)
             "item_rating", "item_rating_reviews_count", "item_price",
             "item_is_phone_hidden", "item_is_message_forbidden"]
TRAIN_COLS = ["search_query", "search_location_id", "search_infm_params_text",
              "item_id", "item_location_id", "item_latitude", "item_longitude", "item_microcat_id"]


def _coords(df: pd.DataFrame) -> pd.DataFrame:
    """Координаты в данных строками -> float (как в geo.load_geo_data)."""
    df["lat"] = pd.to_numeric(df["item_latitude"], errors="coerce").astype(float)
    df["lon"] = pd.to_numeric(df["item_longitude"], errors="coerce").astype(float)
    return df.drop(columns=["item_latitude", "item_longitude"])


def load_data(mode: str, limit: int | None = None):
    """Читает корпус, train и запросы режима.

    val   — запросы и ответы отложенной валидации (src/validation.py, 3000 запросов, seed 42),
            обучение и карты только на train[train_mask]: строки отложенных запросов исключены;
    bench — запросы benchmark_queries, обучение на полном train.
    Возвращает (items, train, train_fit, queries, labels | None). Порядок строк корпуса = порядок
    в BM25 и эмбеддингах; порядок строк train исходный (к нему привязана train_mask).
    """
    items = _coords(pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS))
    items["item_id"] = items["item_id"].astype(str)
    train = _coords(pd.read_parquet(DATA / "train.parquet", columns=TRAIN_COLS))
    train["item_id"] = train["item_id"].astype(str)
    if mode == "val":
        queries, labels, mask = load_validation()
        # маска привязана к порядку строк train.parquet: проверяем, что она исключает ровно отложенные запросы
        keys = set(pd.read_parquet(WORK / "val_keys.parquet")["query_key"])
        held = add_query_key(train[["search_query", "search_location_id", "search_infm_params_text"]])
        assert len(train) == len(mask) and (held["query_key"].isin(keys).to_numpy() == ~mask).all(), \
            "train_mask не соответствует train.parquet"
        train_fit = train[mask]
    else:
        queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
        labels, train_fit = None, train
    queries["query_id"] = queries["query_id"].astype(str)
    if limit:                                   # быстрая проверка на первых N запросах
        queries = queries.head(limit).reset_index(drop=True)
        if labels is not None:
            labels = {q: labels[q] for q in queries["query_id"]}
    print(f"корпус {len(items)}, train {len(train)} (для обучения {len(train_fit)}), запросов {len(queries)}")
    return items, train, train_fit, queries, labels


# ================================================================== шаг 1: BM25 (офлайн, кэш)

def build_sparse_index(items: pd.DataFrame, cfg: Config) -> LemmaIndex:
    """BM25 по леммам и стемам.

    Зачем: 96% выбранных объявлений делят с запросом хотя бы одно слово (eda/00_summary.md), лексика —
    сильнейший одиночный сигнал (BM25 в гео-пуле 0.745). Лемма+стем против одного стема: +0.007 у BM25
    (27 запросов лучше / 4 хуже), в гибриде +0.001 — почти шум, но не хуже (eda/09_lemma.md).
    Индекс зависит только от корпуса, поэтому общий для валидации и бенчмарка и кэшируется в work/pipeline/.
    Ключ кэша: параметры BM25, id объявлений корпуса и исходники text/sparse/sparse_lemma.
    """
    key = fingerprint(cfg.sparse_scheme, cfg.sparse_variant, cfg.bm25_k1, cfg.bm25_b,
                      "\n".join(items["item_id"]).encode(),
                      code_hash("text.py", "sparse.py", "sparse_lemma.py"))
    path = OUT / f"bm25_{cfg.sparse_scheme}_{cfg.sparse_variant}.pkl"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get("key") == key:
        with open(path, "rb") as f:
            idx = pickle.load(f)
        print(f"BM25: из кэша {path.relative_to(ROOT)}")
        return idx
    print("BM25: строю индекс (~1.5 мин на CPU, пик ~5 ГБ RAM)")
    idx = LemmaIndex.build(items, scheme=cfg.sparse_scheme, variant=cfg.sparse_variant,
                           k1=cfg.bm25_k1, b=cfg.bm25_b)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(json.dumps({"key": key, "scheme": cfg.sparse_scheme, "variant": cfg.sparse_variant,
                                     "k1": cfg.bm25_k1, "b": cfg.bm25_b, "n_items": len(items),
                                     "build_sec": round(idx.build_sec, 1)}, ensure_ascii=False, indent=1))
    return idx


# ================================================================== шаги 2–3: dense

def dense_device_warning(cfg: Config) -> None:
    """Без GPU пайплайн работает, но dense-часть медленнее, а без кэша корпуса — очень медленно."""
    if torch.cuda.is_available():
        return
    warnings.warn(
        "GPU не найден: bge-m3 считается на CPU в fp32. Запросы кодируются за минуты, результат может "
        "чуть отличаться от GPU-fp16 (порядок близких кандидатов). Эмбеддинги корпуса берутся из "
        f"work/emb/{cfg.dense_tag}.npy; если их нет, кодирование 189k документов на CPU займёт часы "
        "(на RTX 3070 ~20 мин) — посчитайте их на машине с GPU: "
        f"python src/dense.py encode {cfg.dense_model} --max-seq {cfg.dense_max_seq} --batch 64 --tag {cfg.dense_tag}",
        stacklevel=2)


def load_corpus_embeddings(items: pd.DataFrame, cfg: Config, allow_cpu_encode: bool = False) -> np.ndarray:
    """Эмбеддинги корпуса bge-m3 (float16, [189212, 1024]) из кэша work/emb/ или кодирование заново.

    Зачем dense: ловит синонимы и перефразы, которых нет в BM25 («расхламление» -> «вывоз хлама»).
    bge-m3 в гео-пуле одна даёт 0.764 — лучше e5-small/e5-base/USER-base (eda/07_dense.md), гибрид
    с BM25 +0.058 к одному BM25 (eda/08_hybrid.md). Кэш проверяется по параметрам кодирования
    (модель, длина, префикс, размер) и по порядку item_id — он должен совпадать с порядком корпуса.
    """
    spec = MODELS[cfg.dense_model]
    meta_path = EMB / f"{cfg.dense_tag}.json"
    ok = meta_path.exists() and (EMB / f"{cfg.dense_tag}.npy").exists()
    if ok:
        meta = json.loads(meta_path.read_text())
        ok = (meta.get("model") == spec["name"] and meta.get("max_seq") == cfg.dense_max_seq
              and meta.get("d_prefix") == spec["d"] and meta.get("n") == len(items))
        if not ok:
            print(f"кэш {meta_path.name} не соответствует параметрам: {meta}")
    if ok:
        emb, ids = load_corpus_emb(cfg.dense_tag)
        assert (ids.astype(str) == items["item_id"].to_numpy()).all(), "порядок item_id в кэше не совпадает с корпусом"
        print(f"эмбеддинги корпуса: из кэша work/emb/{cfg.dense_tag}.npy, {emb.shape}, "
              f"посчитаны на {meta.get('device')} ({meta.get('precision')}) за {meta.get('seconds_this_run')} с")
        return emb
    if not torch.cuda.is_available() and not allow_cpu_encode:
        sys.exit("нет кэша эмбеддингов корпуса и нет GPU: кодирование на CPU займёт часы. "
                 "Скопируйте work/emb/ с машины с GPU или запустите с --allow-cpu-encode.")
    print("эмбеддинги корпуса: кодирую (GPU RTX 3070: ~20 мин, пик ~3 ГБ VRAM при batch 64)")
    emb, ids = encode_corpus(cfg.dense_model, cfg.dense_max_seq, batch_size=64, tag=cfg.dense_tag,
                             items=items[["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]])
    return emb


def encode_query_embeddings(queries: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Эмбеддинги запросов bge-m3 (float32, нормированы). Запрос = текст + значения фильтров в скобках.

    Модель после кодирования выгружается, чтобы в видеопамяти одновременно были либо веса модели (~1.1 ГБ),
    либо эмбеддинги корпуса (~0.4 ГБ), но не оба (GPU делится с другими задачами).
    """
    if torch.cuda.is_available():
        # dense.load_model грузит веса на GPU в fp32 (2.3 ГБ) и потом переводит в fp16. Здесь веса сразу
        # читаются в fp16 (эмбеддинги побитово те же, проверено), пик видеопамяти ~1.1 ГБ вместо 2.3.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODELS[cfg.dense_model]["name"], device="cpu",
                                    model_kwargs={"dtype": torch.float16}).to("cuda")
        model.max_seq_length = cfg.dense_max_seq
    else:
        # на CPU без int8-квантизации: она теряет ~0.03 R@50 (eda/07_dense.md), а запросов всего тысячи
        model = load_model(cfg.dense_model, cfg.dense_max_seq, quantize=False)
    q = encode_queries(model, cfg.dense_model, queries, with_filters=cfg.dense_query_filters,
                       batch_size=cfg.dense_batch)
    del model
    gc.collect()                    # без сборки мусора веса модели остаются в видеопамяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return q


# ================================================================== шаг 4: гео-карты (по train)

def build_geo_maps(items: pd.DataFrame, train: pd.DataFrame, train_fit: pd.DataFrame, regions: set) -> GeoMaps:
    """Карты локаций для гео-пула запроса.

    Зачем: локация объявления совпадает с локацией поиска в 83% пар (93% у городских запросов),
    а 17% запросов ищут по региону или всей России (eda/00_summary.md). Поиск по всему корпусу даёт 0.41,
    поиск в своём городе — 0.74 (eda/06_sparse.md).
    Центроиды локаций — по координатам объявлений корпуса и train (атрибуты, не ответы).
    Доли «регион -> город» и пары «город поиска -> город объявления» — только по train_fit
    (train[train_mask] для валидации, полный train для бенчмарка), иначе утечка ответов.
    Город из текста запроса (CityFinder) не нужен: правила с ним дали ±0 (eda/10_geo.md), и выбранные
    мягкие правила его не используют.
    """
    corpus = items[["item_id", "item_location_id", "lat", "lon"]]
    cent = centroids(corpus, train)
    return GeoMaps.build(corpus, cent, regions, None, train_fit)


def geo_rule(loc: int, regions: set, cfg: Config) -> str:
    """Гео-правило запроса по типу локации поиска (город / регион / вся Россия)."""
    if loc == RUSSIA:
        return cfg.geo_russia_rule
    if loc in regions:
        return cfg.geo_region_rule
    return cfg.geo_city_rule


# ================================================================== шаги 5–6: ретривал + RRF + гео

def retrieve_candidates(queries: pd.DataFrame, sparse: LemmaIndex, corpus_emb: np.ndarray, q_emb: np.ndarray,
                        maps: GeoMaps, item_ids: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Топ-300 кандидатов на запрос: BM25 и dense внутри гео-пула, слияние RRF + гео-бонус.

    Для каждого запроса:
      1. скоры BM25 и косинусы bge-m3 по всему корпусу (dense — матрица батч × корпус на GPU);
      2. гео-пул по правилу (geo_rule): город — свой город + соседи ≤100 км + частые пары из train,
         регион — все города, куда из него ходят в train; каждый источник берёт свои топ-300 внутри пула
         и добирает из всего корпуса, если в пуле мало (tiered_top);
      3. RRF k=60: score = Σ 1/(60 + rank) по двум спискам. RRF не зависит от шкалы скоров и почти
         не зависит от k; все разумные слияния дали 0.770–0.776, RRF выбран за устойчивость (eda/08_hybrid.md);
      4. мягкий гео-бонус к RRF: +0.015 своему городу, +0.02 крупным городам региона (≥1% выборов),
         +0.005 для «всей России». Мягкий бонус лучше жёстких ярусов: гибрид 0.803 -> 0.858,
         регионы 0.53 -> 0.77 (eda/10_geo.md). Максимум RRF одного источника 1/61 ≈ 0.016,
         то есть бонус своего города ≈ «ещё один источник на первом месте».
    Возвращает DataFrame(query_id, item_id str, score float32, rank int32 с 1), ровно n_cands на запрос.
    """
    K, bs = cfg.n_cands, cfg.dense_batch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = torch.from_numpy(corpus_emb).to(dev, torch.float16 if dev == "cuda" else torch.float32)
    locs = queries["search_location_id"].to_numpy()
    texts = queries["search_query"].to_numpy()
    all_pos = np.arange(len(item_ids))
    pos_out, score_out = [], []
    t0 = time.time()
    for b0 in range(0, len(queries), bs):
        with torch.no_grad():
            dense_scores = (torch.from_numpy(q_emb[b0:b0 + bs]).to(dev, D.dtype) @ D.T).float().cpu().numpy()
        for j, dv in enumerate(dense_scores):
            i = b0 + j
            loc = int(locs[i])
            ssc = sparse.scores(texts[i])
            # глобальные топы для добора, если в гео-пуле меньше K кандидатов (BM25 — только скор > 0)
            g_s = SparseIndex._top(all_pos, ssc, 2 * K)
            g_d = np.argpartition(-dv, 2 * K)[:2 * K]
            g_d = g_d[np.argsort(-dv[g_d], kind="stable")]
            rule = geo_rule(loc, maps.regions, cfg)
            pool = maps.tiers(maps.base_rule(rule), loc)          # один плоский ярус — гео-пул
            top_sparse, _ = tiered_top(ssc, pool, K, g_s, positive=True)
            top_dense, _ = tiered_top(dv, pool, K, g_d, positive=False)
            bonus_mask, bonus_w = maps.bonus_mask(rule, loc)
            fused, fsc = rrf([top_sparse, top_dense], k_rrf=cfg.rrf_k, n_out=K, bonus=bonus_mask, bonus_w=bonus_w)
            pos_out.append(fused)
            score_out.append(fsc)
        if (b0 // bs) % 5 == 0:
            print(f"  {min(b0 + bs, len(queries))}/{len(queries)} за {time.time() - t0:.0f} с", flush=True)
    del D
    if dev == "cuda":
        torch.cuda.empty_cache()
    n = [len(p) for p in pos_out]
    cands = pd.DataFrame({
        "query_id": np.repeat(queries["query_id"].to_numpy(), n),
        "item_id": item_ids[np.concatenate(pos_out)].astype(str),
        # float32 — как в сохранённых кандидатах экспериментов: бонусы дальше считаются от этого значения
        "score": np.concatenate(score_out).astype(np.float32),
        "rank": np.concatenate([np.arange(1, m + 1) for m in n]).astype(np.int32)})
    assert (np.array(n) == K).all(), "у части запросов меньше n_cands кандидатов"
    return cands


# ================================================================== шаг 7: бонусы метаданных

def apply_meta_bonus(cands: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame,
                     train_fit: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Переупорядочивание топ-300 бонусами за метаданные (новых id не добавляет).

    score' = score + 0.003·[совпал «Вид услуги»] + 0.0015·[совпал «Тип услуги»]
                   + 0.007·P(микрокатегория объявления | запрос) + 0.002·доля его микрокатегории в топ-20.
    Зачем: ~85% выборов по тексту приходятся на одну микрокатегорию, «Вид/Тип услуги» из фильтра
    совпадает у 97% ответов (eda/00_summary.md). Жёсткий фильтр по ним вредит (микрокатегория как фильтр
    −15%, «жёстко с запасом» хуже мягкого бонуса), поэтому только бонус.
    Классификатор: TF-IDF слов 1–2 и символов 3–5 + SGD-логрегрессия, учится на train_fit (~30 с), top-1 0.71.
    Веса выбраны на половине A валидации, на B +0.011 [+0.004; +0.018], на всех 0.858 -> 0.869 (eda/11_meta_boost.md).
    База — score с гео-бонусом (base="score"): ранговая база сжимает гео-разрыв и проваливается при больших весах.
    """
    t = time.time()
    model = rm.fit_microcat(train_fit)          # seed=0 внутри: детерминированно
    print(f"  классификатор микрокатегорий: {len(model.classes_)} классов, {time.time() - t:.0f} с")
    f = rm.filter_match(cands, queries, items)
    f = rm.microcat_prob(f, queries, items, model)
    f["prf"] = rm.prf_microcat(f, items, top=cfg.meta_prf_top).to_numpy()
    return rm.rerank_bonus(f, base="score", **cfg.meta_weights)


# ================================================================== шаг 8: опциональные надстройки

def apply_quality_bonus(cands: pd.DataFrame, items: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """ОПЦИОНАЛЬНО (--quality): мягкий бонус за число отзывов и длину описания (eda/15_quality.md).

    Признаки переводятся во внутризапросный перцентиль среди 300 кандидатов, вес 0.0025 / 0.0019:
    лучший по отзывам кандидат получает на 0.0025 больше худшего. Валидация 0.869 -> 0.876, но прирост
    в основном на ответах «с историей» (отзывы ~ популярность); на срезе без истории (≈ бенчмарк) +0.004 в шуме.
    """
    import rerank_quality as rq
    return rq.rerank_quality(cands, items[rq.QUALITY_COLS], cfg.quality_weights)


def apply_extra_dense(*_):
    """НЕ ВКЛЮЧЕНО (--extra-dense qwen3): третий список в RRF от Qwen3-Embedding-0.6B (experiments/dense_qwen.py).
    Объединение кандидатов растёт (0.926 -> 0.934), но Recall@50 после слияния не растёт (eda/13_qwen_ensemble.md)."""
    raise NotImplementedError("шаг --extra-dense не включён: прироста нет (eda/13_qwen_ensemble.md)")


def apply_cross_encoder(*_):
    """НЕ ВКЛЮЧЕНО (--rerank-ce): bge-reranker-v2-m3 по топ-100 + RRF с исходным рангом (experiments/rerank_ce.py).
    На 800 запросах +0.008 [−0.003; +0.019] — в шуме, ~1 с/запрос на GPU, 80 с на CPU (eda/12_rerank.md)."""
    raise NotImplementedError("шаг --rerank-ce не включён: прирост в шуме (eda/12_rerank.md)")


# ================================================================== итоговая схема: дообученный эмбеддер + LTR

def load_ft_corpus_embeddings(items: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Эмбеддинги корпуса дообученной bge-m3-lora-all (float16, [189212, 1024]) из кэша work/emb/.

    Зачем: дообучение на парах «запрос -> выбранное объявление» из train учит модель языку запросов Авито
    («сантехник срочно» -> объявления с «аварийный выезд»). Одна модель вместо bge-m3 в схеме meta: 0.869 -> 0.885
    (eda/17_finetune.md). Кодирование корпуса: python src/ft_eval.py encode --name bge-m3-lora-all --batch 64
    (~19 мин на RTX 3070)."""
    meta_path, npy = EMB / f"{cfg.ft_tag}.json", EMB / f"{cfg.ft_tag}.npy"
    if not (meta_path.exists() and npy.exists()):
        sys.exit(f"нет кэша work/emb/{cfg.ft_tag}.npy: python src/ft_eval.py encode --name {cfg.ft_model} --batch 64")
    meta = json.loads(meta_path.read_text())
    assert meta.get("model", "").endswith(cfg.ft_model) and meta.get("max_seq") == 256 and meta.get("n") == len(items), \
        f"кэш {meta_path.name} не соответствует модели {cfg.ft_model}: {meta}"
    emb, ids = load_corpus_emb(cfg.ft_tag)
    assert (ids.astype(str) == items["item_id"].to_numpy()).all(), "порядок item_id в кэше не совпадает с корпусом"
    print(f"эмбеддинги корпуса {cfg.ft_model}: из кэша work/emb/{cfg.ft_tag}.npy, {emb.shape}")
    return emb


def encode_ft_query_embeddings(queries: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Эмбеддинги запросов дообученной моделью (текст + значения фильтров), модель потом выгружается."""
    import ltr_ft
    if not (WORK / "models" / cfg.ft_model / "train_info.json").exists():
        sys.exit(f"нет дообученной модели work/models/{cfg.ft_model}: python src/ft_train.py --model BAAI/bge-m3 "
                 f"--name {cfg.ft_model} --q-prefix \"\" --d-prefix \"\" --lora 16 --lr 1e-4 --grad-ckpt --batch 128 "
                 "--max-pairs 0 --pretok")
    return ltr_ft.encode_ft_queries(cfg.ft_model, queries)


def retrieve_union(queries, sparse, corpus_emb, ft_emb, q_emb, q_ft, maps, item_ids, cfg) -> pd.DataFrame:
    """Кандидаты итоговой схемы: geo2 топ-300 (как retrieve_candidates, побитово) ∪ топ-100 дообученной
    модели в том же гео-пуле. Добор новых кандидатов: объединение топ-300 geo2 и топ-100 bge-m3-lora-all даёт
    потолок 0.942 -> 0.948; сам по себе добор почти ничего не даёт (+0.001), но вместе с признаком
    дообученной модели ранжировщик ставит их наверх (+0.012 к LTR без обоих, eda/19_ltr_ft.md).
    Возвращает кандидатов со скорами источников (колонки ltr_ft.CAND_COLS)."""
    import ltr_ft
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    D, Dft = torch.from_numpy(corpus_emb).to(dev, dt), torch.from_numpy(ft_emb).to(dev, dt)
    c = ltr_ft.union_candidates(queries, q_emb, q_ft, sparse, D, Dft, maps, item_ids, cfg, tag="запросы")
    del D, Dft
    if dev == "cuda":
        torch.cuda.empty_cache()
    return c


def apply_ltr(union: pd.DataFrame, queries: pd.DataFrame, train: pd.DataFrame, train_fit: pd.DataFrame,
              regions: set, mode: str, cfg: Config) -> pd.DataFrame:
    """Признаки и ранжирование 5 бустерами LightGBM lambdarank (среднее z-оценок внутри запроса) -> топ-300.

    Бустеры учились на ~21 тыс. запросах пула из train[train_mask] (src/ltr_big.py sample), отдельно от
    валидации; для запроса пула всё, что строится по train (классификатор, гео-доли, дообученный эмбеддер),
    строилось без строк его фолда (5 фолдов по тексту), поэтому признаки на обучении и здесь одинаково честные.
    Признаки строятся теми же функциями, что при обучении: объекты — по train_fit режима
    (val — train[train_mask], bench — полный train), флаг «текст запроса есть в train» — по нему же."""
    import rerank_meta as rm
    import ltr_ft
    from ltr import GeoCtx, _load_items, to_cands
    from validation import add_flags, normalize_text
    items = _load_items()
    q = queries
    if mode == "bench":                          # у валидации флаги уже посчитаны по train[train_mask]
        q = add_flags(queries, set(normalize_text(train_fit["search_query"])), regions)
    corpus = items[["item_id", "item_location_id"]].copy()
    corpus["lat"] = pd.to_numeric(items["item_latitude"], errors="coerce").astype(float)
    corpus["lon"] = pd.to_numeric(items["item_longitude"], errors="coerce").astype(float)
    cent = centroids(corpus, train)
    t = time.time()
    mc = rm.fit_microcat(train_fit)
    print(f"  классификатор микрокатегорий: {len(mc.classes_)} классов, {time.time() - t:.0f} с")
    geo = GeoCtx(train_fit, cent, regions)
    f = ltr_ft.features_for(union, q, items, mc, geo)
    p = ltr_ft.ltr_predict(f, ROOT / cfg.ltr_models)
    c = to_cands(f, p)
    return c[c["rank"] <= cfg.n_cands].reset_index(drop=True)


# ================================================================== оценка и ответ

def top_k_lists(cands: pd.DataFrame, k: int) -> dict:
    """query_id -> первые k item_id по rank."""
    top = cands[cands["rank"] <= k].sort_values(["query_id", "rank"], kind="mergesort")
    return top.groupby("query_id", sort=False)["item_id"].agg(list).to_dict()


def evaluate(cands: pd.DataFrame, queries: pd.DataFrame, labels: dict, name: str) -> pd.Series:
    """Recall@50 (метрика задачи), кривая @100/@300 и срезы валидации (eda/05_validation.md)."""
    r50 = recall_at_k(top_k_lists(cands, 50), labels, 50)
    curve = {k: recall_at_k(top_k_lists(cands, k), labels, k).mean() for k in (100, 300)}
    print(f"[{name}] Recall@50 = {r50.mean():.4f}  (@100 {curve[100]:.4f}, @300 {curve[300]:.4f}; "
          f"±{r50.std() / np.sqrt(len(r50)):.4f} — std/√n)")
    return r50


def print_slices(r50: pd.Series, queries: pd.DataFrame) -> None:
    """Срезы Recall@50 валидации (eda/05_validation.md) + регионы отдельно от «всей России»."""
    sl = slices(r50, queries)
    q = queries.set_index("query_id").loc[r50.index]
    russia = (q["search_location_id"] == RUSSIA).to_numpy()
    reg = q["is_region"].to_numpy()
    sl.loc["— регионы без 621540"] = [int((reg & ~russia).sum()), r50[reg & ~russia].mean()]
    sl.loc["— вся Россия (621540)"] = [int(russia.sum()), r50[russia].mean()]
    sl["запросов"] = sl["запросов"].astype(int)
    print(sl.round(4).to_string())


def compare_with_reference(cands: pd.DataFrame, mode: str, stage: str, k: int = 50) -> None:
    """Сверка с результатом прежних экспериментов: доля запросов с тем же набором топ-k и среднее пересечение."""
    path = REFERENCE.get((mode, stage))
    if path is None or not path.exists():
        return
    mine = {q: set(v) for q, v in top_k_lists(cands, k).items()}
    if path.suffix == ".csv":
        ref_df = pd.read_csv(path, dtype=str, keep_default_na=False)
        ref = {q: set(a.split()) for q, a in zip(ref_df["query_id"], ref_df["answer"])}
    else:
        ref = {q: set(v) for q, v in top_k_lists(pd.read_parquet(path), k).items()}
    common = [q for q in mine if q in ref]
    same = np.mean([mine[q] == ref[q] for q in common])
    overlap = np.mean([len(mine[q] & ref[q]) / max(len(ref[q]), 1) for q in common])
    print(f"  сверка с {path.relative_to(ROOT)}: запросов {len(common)}, топ-{k} совпадает целиком у {same:.2%}, "
          f"среднее пересечение {overlap:.2%}")


def write_and_check_answer(cands: pd.DataFrame, queries: pd.DataFrame, out: Path, k: int) -> None:
    """answer.csv: query_id,answer (k item_id через пробел), затем проверка формата check_answer.py."""
    preds = top_k_lists(cands, k)
    assert all(len(preds[q]) == k for q in queries["query_id"]), "в ответе не ровно k id"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_answer(preds, out, queries)
    print(f"записан {out}")
    res = subprocess.run([sys.executable, str(ROOT / "check_answer.py"), str(out), "--data", str(DATA)],
                         capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        sys.exit(f"check_answer.py нашёл ошибки формата:\n{res.stderr}")


# ================================================================== main

def parse_args() -> tuple[argparse.Namespace, Config]:
    """Аргументы командной строки -> (args, Config выбранной схемы)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["val", "bench"], required=True,
                    help="val — train[train_mask] и метрика на валидации; bench — полный train и answer.csv")
    ap.add_argument("--out", default=str(ROOT / "answer.csv"), help="куда писать ответ (режим bench)")
    ap.add_argument("--limit", type=int, default=None, help="только первые N запросов (быстрая проверка)")
    ap.add_argument("--allow-cpu-encode", action="store_true",
                    help="разрешить кодировать корпус на CPU при отсутствии кэша (часы)")
    ap.add_argument("--scheme", choices=["ltr", "meta"], default="ltr",
                    help="ltr — итоговая (дообученный эмбеддер + LightGBM, сабмит 4); "
                         "meta — geo2 + ручные бонусы (сабмит 1)")
    ap.add_argument("--quality", action="store_true", help="схема meta: мягкий бонус качества (eda/15_quality.md)")
    ap.add_argument("--extra-dense", choices=["qwen3"], default=None, help="НЕ ВКЛЮЧЕНО: второй эмбеддер в RRF")
    ap.add_argument("--rerank-ce", action="store_true", help="НЕ ВКЛЮЧЕНО: cross-encoder по топ-100")
    ap.add_argument("--max-vram-gb", type=float, default=2.0, help="потолок видеопамяти процесса")
    a = ap.parse_args()
    if a.quality and a.scheme == "ltr":
        ap.error("--quality — только для --scheme meta (в итоговой схеме качество — признаки ранжировщика)")
    cfg = Config(quality=a.quality, extra_dense=a.extra_dense, ltr=a.scheme == "ltr",
                 rerank_ce=a.rerank_ce, max_vram_gb=a.max_vram_gb)
    return a, cfg


def main() -> None:
    """Шаги пайплайна по порядку с замерами времени и памяти; в конце метрика (val) или answer.csv (bench)."""
    a, cfg = parse_args()
    set_seed(cfg.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.max_vram_gb / total))
    dense_device_warning(cfg)
    # заглушки падают сразу, до тяжёлых шагов
    if cfg.extra_dense:
        apply_extra_dense()
    if cfg.rerank_ce:
        apply_cross_encoder()

    if cfg.ltr:
        need = [ROOT / cfg.ltr_models / f for f in ["ltr.json"] + [f"ltr_{k}.txt" for k in range(5)]]
        missing = [str(x.relative_to(ROOT)) for x in need if not x.exists()]
        if missing:
            sys.exit(f"нет бустеров {missing}: обучение — bash train_all.sh (README.md «Как воспроизвести»)")

    with step("0. данные"):
        items, train, train_fit, queries, labels = load_data(a.mode, a.limit)
        regions = region_location_ids()
        item_ids = items["item_id"].to_numpy()
    with step("1. BM25 lemma+stem (офлайн, кэш)"):
        sparse = build_sparse_index(items, cfg)
    # запросы кодируются до загрузки корпусов на GPU: в видеопамяти либо веса модели, либо корпуса
    with step("3. эмбеддинги запросов bge-m3"):
        q_emb = encode_query_embeddings(queries, cfg)
    if cfg.ltr:
        with step("3б. эмбеддинги запросов дообученной bge-m3-lora-all"):
            q_ft = encode_ft_query_embeddings(queries, cfg)
    with step("2. эмбеддинги корпуса bge-m3 (офлайн, кэш)"):
        corpus_emb = load_corpus_embeddings(items, cfg, a.allow_cpu_encode)
    with step("4. гео-карты по train"):
        maps = build_geo_maps(items, train, train_fit, regions)
    if cfg.ltr:
        with step("2б. эмбеддинги корпуса bge-m3-lora-all (офлайн, кэш)"):
            ft_emb = load_ft_corpus_embeddings(items, cfg)
        with step("5. ретривал: BM25 + bge-m3 в гео-пуле, RRF + гео-бонус; + топ-100 bge-m3-lora-all"):
            union = retrieve_union(queries, sparse, corpus_emb, ft_emb, q_emb, q_ft, maps, item_ids, cfg)
        del sparse, corpus_emb, ft_emb
        # geo2 (топ-300 без новых кандидатов) — для сверки с прежними экспериментами
        stages = {"hybrid": union[union["is_new"] == 0][["query_id", "item_id", "score", "rank"]]}
        with step("6+7. признаки и ранжирование 5 бустерами LightGBM"):
            cands = apply_ltr(union, queries, train, train_fit, regions, a.mode, cfg)
        stages["ltr"] = cands
    else:
        with step("5. ретривал: BM25 + dense в гео-пуле, RRF + гео-бонус"):
            cands = retrieve_candidates(queries, sparse, corpus_emb, q_emb, maps, item_ids, cfg)
        del sparse, corpus_emb
        stages = {"hybrid": cands}
        with step("6. бонусы метаданных (классификатор микрокатегорий по train)"):
            cands = apply_meta_bonus(cands, queries, items, train_fit, cfg)
        stages["meta"] = cands
        if cfg.quality:
            with step("7. бонус качества (опционально)"):
                cands = apply_quality_bonus(cands, items, cfg)
            stages["quality"] = cands
    scheme = "ltr" if cfg.ltr else "meta"
    cands.to_parquet(OUT / f"cands_{a.mode}_{scheme}.parquet", index=False)

    print("\n=== результат")
    if a.mode == "val":
        for name, c in stages.items():
            r = evaluate(c, queries, labels, name)
            compare_with_reference(c, "val", name)
        print_slices(r, queries)
        r.rename("recall@50").to_frame().to_parquet(OUT / f"val_per_query_{scheme}.parquet")
    else:
        for name, c in stages.items():
            print(f"[{name}]")
            compare_with_reference(c, "bench", name)
        write_and_check_answer(cands, queries, Path(a.out), cfg.n_answer)

    tm = pd.DataFrame(TIMINGS)
    tm.loc[len(tm)] = {"шаг": "всего", "сек": round(time.time() - t_all, 1),
                       "пик RAM процесса, ГБ": tm["пик RAM процесса, ГБ"].max(),
                       "пик VRAM шага, ГБ": tm["пик VRAM шага, ГБ"].max()}
    device = "gpu" if torch.cuda.is_available() else "cpu"
    tm.to_csv(OUT / f"timings_{a.mode}_{scheme}_{device}.csv", index=False)
    (OUT / f"config_{a.mode}_{scheme}.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=1))
    print("\n=== время и ресурсы (устройство: " + ("GPU " + torch.cuda.get_device_name(0)
                                                   if torch.cuda.is_available() else "CPU") + ")")
    print(tm.to_string(index=False))


if __name__ == "__main__":
    main()
