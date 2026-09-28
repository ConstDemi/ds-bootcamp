"""Валидация, похожая на бенчмарк, и метрика Recall@K.

Как устроен бенчмарк (см. eda/01_queries.md):
- «запрос» = нормализованный текст + локация + строка фильтров;
- похоже, на каждый уникальный текст взят один запрос, тексты отобраны равномерно,
  а остальные запросы с тем же текстом остались в train;
- все правильные ответы лежат в корпусе benchmark_items.

Валидация повторяет эту схему на train:
1. склеиваем строки train в запросы (query_key), повторные выборы схлопываются;
2. оставляем запросы, у которых хотя бы одно выбранное объявление есть в корпусе,
   и считаем правильными ответами только эти объявления (как в бенчмарке);
3. на каждый текст берём один случайный такой запрос;
4. отбираем тексты по слоям (текст встречается в других запросах train × есть фильтр)
   так, чтобы доли слоёв совпали с бенчмарком;
5. строки отложенных запросов исключаем из обучения (train_mask); другие запросы
   того же текста остаются.

Флаги запросов (text_in_train, has_filter, is_region, target_has_history) считаются
один раз функцией add_flags и хранятся в таблице. Для валидации «текст есть в train»
считается по train без отложенных строк, для бенчмарка — по полному train.
describe_queries и slices только читают эти колонки, поэтому перепутать нельзя.

Использование:
    from validation import load_validation, recall_at_k, slices
    queries, labels, train_mask = load_validation()   # файлы из work/
    per_query = recall_at_k(preds, labels)             # preds: query_id -> список item_id
    print(per_query.mean()); print(slices(per_query, queries))
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
WORK = Path(__file__).resolve().parent.parent / "work"

SEARCH_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
               "search_infm_params_text", "search_category"]

_WS = re.compile(r"\s+")


def normalize_text(s: pd.Series) -> pd.Series:
    """Нижний регистр, ё→е, одиночные пробелы: «Баня  на Дровах» и «баня на дровах» — один текст."""
    return (s.fillna("").astype(str).str.lower().str.replace("ё", "е", regex=False)
            .str.replace(_WS, " ", regex=True).str.strip())


def add_query_key(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет text_norm и query_key = текст | локация | фильтры (флаг доставки почти пуст, не учитываем)."""
    df = df.copy()
    df["text_norm"] = normalize_text(df["search_query"])
    df["query_key"] = (df["text_norm"] + "|" + df["search_location_id"].astype(str) + "|"
                       + df["search_infm_params_text"].fillna("").astype(str))
    return df


def has_filter(q: pd.DataFrame) -> pd.Series:
    """Есть непустая строка фильтров search_infm_params_text."""
    return q["search_infm_params_text"].fillna("").str.strip().ne("")


def load_train() -> pd.DataFrame:
    """Поисковые колонки train + item_id + text_norm/query_key. Остальные колонки не читаем (память)."""
    return add_query_key(pd.read_parquet(DATA / "train.parquet", columns=SEARCH_COLS + ["item_id"]))


def region_location_ids() -> set:
    """search_location_id, которых нет ни у одного объявления: регионы и «вся Россия»."""
    tr = pd.read_parquet(DATA / "train.parquet", columns=["search_location_id", "item_location_id"])
    item_locs = set(tr["item_location_id"]) | set(
        pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_location_id"])["item_location_id"])
    search_locs = set(tr["search_location_id"]) | set(
        pd.read_parquet(DATA / "benchmark_queries.parquet", columns=["search_location_id"])["search_location_id"])
    return search_locs - item_locs


def add_flags(q: pd.DataFrame, train_texts: set, regions: set) -> pd.DataFrame:
    """Флаги для слоёв и срезов.

    train_texts — тексты того train, на котором учится модель:
    для валидации — train[train_mask], для бенчмарка — весь train.
    """
    q = q.copy()
    if "text_norm" not in q:
        q["text_norm"] = normalize_text(q["search_query"])
    q["text_in_train"] = q["text_norm"].isin(train_texts)
    q["has_filter"] = has_filter(q)
    q["is_region"] = q["search_location_id"].isin(regions)
    return q


def build_validation(n_queries: int = 3000, seed: int = 42, train: pd.DataFrame | None = None,
                     regions: set | None = None):
    """Возвращает (val_queries, val_labels, train_mask).

    val_queries — таблица в формате benchmark_queries: query_id + search_* + text_norm, query_key
                  и флаги text_in_train, has_filter, is_region, target_has_history;
    val_labels  — dict query_id -> set(item_id) правильных ответов из корпуса;
    train_mask  — bool-массив длины train: True для строк, на которых можно учиться.
    train и regions можно передать готовыми (load_train(), region_location_ids()), чтобы не читать файлы заново.
    """
    train = load_train() if train is None else train
    regions = region_location_ids() if regions is None else regions
    corpus_ids = set(pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])["item_id"])
    in_corpus = train["item_id"].isin(corpus_ids)

    # Кандидаты в отложенные: запросы, где хотя бы один выбор есть в корпусе.
    eligible = train.loc[in_corpus, ["text_norm", "query_key", "search_infm_params_text"]] \
        .drop_duplicates("query_key")
    # По одному случайному запросу на текст: в бенчмарке тексты не повторяются.
    per_text = eligible.sample(frac=1.0, random_state=seed).drop_duplicates("text_norm")

    # Слой отложенного запроса. seen: у текста есть другие запросы, т.е. после исключения
    # отложенного запроса текст останется в обучающем train.
    queries_per_text = train.groupby("text_norm")["query_key"].nunique()
    per_text["seen"] = per_text["text_norm"].map(queries_per_text).gt(1)
    per_text["has_filter"] = has_filter(per_text)

    # Доли слоёв в бенчмарке; «текст есть в train» для бенчмарка — по полному train.
    bench = add_flags(pd.read_parquet(DATA / "benchmark_queries.parquet"), set(train["text_norm"]), regions)
    shares = bench[["text_in_train", "has_filter"]].value_counts(normalize=True)
    counts = (shares * n_queries).round().astype(int)
    counts.iloc[0] += n_queries - counts.sum()  # поправка округления, чтобы в сумме было ровно n_queries

    # Условие «ответ есть в корпусе» тянет к частым текстам, поэтому отбираем внутри слоёв.
    parts = []
    for (seen, filt), n in counts.items():
        pool = per_text[(per_text["seen"] == seen) & (per_text["has_filter"] == filt)]
        if len(pool) < n:
            print(f"слой text_in_train={seen}, has_filter={filt}: нужно {n}, есть {len(pool)} — беру все")
        parts.append(pool.sample(n=min(n, len(pool)), random_state=seed))
    val_keys = set(pd.concat(parts)["query_key"])

    # Все строки отложенных запросов уходят из обучения.
    in_val = train["query_key"].isin(val_keys)
    train_mask = ~in_val.to_numpy()

    val_rows = train[in_val]
    labels_by_key = val_rows[in_corpus[in_val]].groupby("query_key")["item_id"].agg(set)
    queries = val_rows.drop_duplicates("query_key").drop(columns=["item_id"]).reset_index(drop=True)
    queries.insert(0, "query_id", [f"val{i:05d}" for i in range(len(queries))])
    val_labels = dict(zip(queries["query_id"], queries["query_key"].map(labels_by_key)))

    # Флаги считаем по обучающей части train (без отложенных строк).
    queries = add_flags(queries, set(train.loc[train_mask, "text_norm"]), regions)
    # История объявления: хотя бы один правильный ответ выбирали в обучающих строках.
    # На бенчмарке так бывает редко, поэтому популярность/история на валидации переоцениваются.
    train_items = set(train.loc[train_mask, "item_id"])
    queries["target_has_history"] = [bool(val_labels[q] & train_items) for q in queries["query_id"]]
    return queries, val_labels, train_mask


def recall_at_k(preds: dict, labels: dict, k: int = 50) -> pd.Series:
    """Recall@K по каждому запросу: |топ-K ∩ правильные| / |правильные|. Среднее — метрика задачи.

    preds: query_id -> список item_id. Запрос без предсказания получает 0.
    Список проверяется так же строго, как answer.csv на платформе: id — строки,
    не больше K штук, без повторов. Иначе ошибка, а не молча завышенная метрика.
    """
    out = {}
    for qid, rel in labels.items():
        top = list(preds.get(qid, []))
        assert len(top) <= k, f"{qid}: {len(top)} id, а можно не больше {k}"
        assert len(set(top)) == len(top), f"{qid}: повторы id в ответе"
        assert all(isinstance(i, str) for i in top), f"{qid}: id должны быть строками"
        out[qid] = len(set(top) & rel) / len(rel)
    return pd.Series(out, name=f"recall@{k}")


def describe_queries(q: pd.DataFrame) -> dict:
    """Признаки, по которым сравниваем валидацию с бенчмарком (q должна пройти add_flags)."""
    return {
        "запросов": len(q),
        "текст есть в train, %": 100 * q["text_in_train"].mean(),
        "слов в среднем": q["text_norm"].str.split().str.len().mean(),
        "без фильтров, %": 100 * (~q["has_filter"]).mean(),
        "регион или вся Россия, %": 100 * q["is_region"].mean(),
        "search_category = 0, %": 100 * (q["search_category"] == 0).mean(),
    }


def slices(per_query: pd.Series, q: pd.DataFrame) -> pd.DataFrame:
    """Recall по срезам: знакомый/новый текст, город/регион, с фильтром/без, история объявления."""
    d = q.set_index("query_id").loc[per_query.index]
    groups = {
        "все": pd.Series(True, index=d.index),
        "текст есть в train": d["text_in_train"],
        "текста нет в train": ~d["text_in_train"],
        "поиск по городу": ~d["is_region"],
        "регион или вся Россия": d["is_region"],
        "с фильтром": d["has_filter"],
        "без фильтра": ~d["has_filter"],
    }
    if "target_has_history" in d:  # есть только у валидации
        groups["у ответа есть история"] = d["target_has_history"]
        groups["ответ без истории"] = ~d["target_has_history"]
    return pd.DataFrame({name: {"запросов": int(m.sum()), "recall": per_query[m].mean()}
                         for name, m in groups.items()}).T


def save_validation(queries: pd.DataFrame, labels: dict, mask: np.ndarray) -> None:
    """Пишет val_queries.parquet, val_labels.parquet (query_id, item_ids) и train_mask.npy в work/."""
    WORK.mkdir(exist_ok=True)
    queries.to_parquet(WORK / "val_queries.parquet", index=False)
    pd.DataFrame({"query_id": list(labels), "item_ids": [sorted(v) for v in labels.values()]}) \
        .to_parquet(WORK / "val_labels.parquet", index=False)
    np.save(WORK / "train_mask.npy", mask)
    # Маска привязана к порядку строк train.parquet. Сохраняем ещё ключи отложенных запросов,
    # чтобы load_validation мог проверить, что маска не съехала.
    pd.Series(sorted(queries["query_key"]), name="query_key").to_frame() \
        .to_parquet(WORK / "val_keys.parquet", index=False)


def load_validation():
    """Читает сохранённую валидацию: (val_queries, val_labels dict, train_mask)."""
    queries = pd.read_parquet(WORK / "val_queries.parquet")
    lab = pd.read_parquet(WORK / "val_labels.parquet")
    labels = {qid: set(items) for qid, items in zip(lab["query_id"], lab["item_ids"])}
    mask = np.load(WORK / "train_mask.npy")
    return queries, labels, mask


def check_mask(train: pd.DataFrame, mask: np.ndarray) -> None:
    """Проверяет, что train прочитан в исходном порядке и маска исключает ровно отложенные запросы.
    train должен содержать query_key (прогнать через add_query_key)."""
    assert len(train) == len(mask), "длина train не совпадает с маской: train прочитан с фильтром?"
    val_keys = set(pd.read_parquet(WORK / "val_keys.parquet")["query_key"])
    held_out = train["query_key"].isin(val_keys).to_numpy()
    assert (held_out == ~mask).all(), "маска съехала: порядок строк train отличается от исходного"


if __name__ == "__main__":
    queries, labels, mask = build_validation()
    save_validation(queries, labels, mask)
    print(f"отложено запросов: {len(queries)}, строк train для обучения: {mask.sum()} из {len(mask)}")
    print(pd.Series(describe_queries(queries)).round(2).to_string())
    print(f"правильных ответов на запрос: {np.mean([len(v) for v in labels.values()]):.2f}; "
          f"у ответа есть история: {100 * queries['target_has_history'].mean():.1f}%")
