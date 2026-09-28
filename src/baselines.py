"""Простые бейзлайны кандидатогенерации.

predict_popular_in_location — «популярное в городе»: для запроса берём объявления,
которые чаще всего выбирали при поиске в той же search_location_id, и добираем
глобальным топом. Функция не читает файлы сама, поэтому одинаково вызывается
и на валидации (train без отложенных строк), и на бенчмарке (полный train).

Запуск из корня репозитория:
    python src/baselines.py            # пишет answer_01_popular.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def popularity_tables(train: pd.DataFrame, corpus_ids, k: int = 50):
    """Считает популярность по строкам train, чьё item_id есть в корпусе.

    Возвращает:
      by_loc      — dict search_location_id -> список до k item_id (по убыванию числа выборов);
      global_top  — список всех корпусных item_id из train по убыванию числа выборов.
    При равенстве числа выборов порядок по item_id — чтобы результат был детерминированным.
    """
    corpus = set(map(str, corpus_ids))
    t = train[["search_location_id", "item_id"]].copy()
    t["item_id"] = t["item_id"].astype(str)
    t = t[t["item_id"].isin(corpus)]

    # глобальный топ: число выборов объявления по всему train
    g = t["item_id"].value_counts().rename("n").reset_index()
    global_top = g.sort_values(["n", "item_id"], ascending=[False, True])["item_id"].tolist()

    # топ внутри локации поиска (работает и для региональных id: это то, что выбирали в регионе)
    c = t.groupby(["search_location_id", "item_id"]).size().rename("n").reset_index()
    c = c.sort_values(["search_location_id", "n", "item_id"], ascending=[True, False, True])
    c = c.groupby("search_location_id").head(k)
    by_loc = c.groupby("search_location_id")["item_id"].agg(list).to_dict()
    return by_loc, global_top


def predict_popular_in_location(queries: pd.DataFrame, train: pd.DataFrame,
                                corpus_ids, k: int = 50) -> dict[str, list[str]]:
    """Бейзлайн «популярное в городе»: query_id -> ровно k item_id из корпуса, без повторов.

    1) топ объявлений по числу выборов в train с тем же search_location_id;
    2) добор глобальным топом популярности;
    3) если и этого мало — добор любыми объявлениями корпуса (по возрастанию item_id).
    """
    by_loc, global_top = popularity_tables(train, corpus_ids, k)
    # общий «хвост» для добора: глобальный топ, затем остальной корпус в фиксированном порядке
    seen = set(global_top[:k])
    fill = global_top[:k] + [i for i in sorted(map(str, corpus_ids)) if i not in seen][:k]

    preds = {}
    for qid, loc in zip(queries["query_id"].astype(str), queries["search_location_id"]):
        out = list(by_loc.get(loc, []))[:k]
        used = set(out)
        for i in fill:  # добираем до k, пропуская то, что уже взяли из локации
            if len(out) >= k:
                break
            if i not in used:
                out.append(i)
                used.add(i)
        preds[qid] = out
    return preds


def write_answer(preds: dict, path, queries: pd.DataFrame | None = None) -> None:
    """Пишет CSV `query_id,answer` (answer — item_id через один пробел), utf-8 без BOM.

    Порядок строк — как в queries (если переданы), иначе как в preds (он и так совпадает
    с порядком queries, если preds получены из predict_popular_in_location).
    """
    order = queries["query_id"].astype(str).tolist() if queries is not None else list(preds)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "answer"])
        for qid in order:
            w.writerow([qid, " ".join(preds[qid])])


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    # читаем только нужные колонки; id — строки (ведущие нули сохраняются)
    train = pd.read_parquet(data / "train.parquet", columns=["search_location_id", "item_id"])
    queries = pd.read_parquet(data / "benchmark_queries.parquet",
                              columns=["query_id", "search_location_id"])
    corpus_ids = pd.read_parquet(data / "benchmark_items.parquet", columns=["item_id"])["item_id"].astype(str)

    K = 50
    preds = predict_popular_in_location(queries, train, corpus_ids, k=K)
    out_path = root / "answer_01_popular.csv"
    write_answer(preds, out_path, queries)
    print("записано:", out_path)

    # статистика: сколько кандидатов пришло из «своей» локации
    by_loc, _ = popularity_tables(train, corpus_ids, K)
    n_local = queries["search_location_id"].map(lambda l: len(by_loc.get(l, []))).clip(upper=K)
    print(f"запросов с >=1 популярным объявлением в своей локации: "
          f"{int((n_local > 0).sum())} из {len(queries)} ({(n_local > 0).mean():.1%})")
    print(f"средняя доля «городских» кандидатов в {K}: {(n_local / K).mean():.3f}")
    print(f"запросов, где локация дала все {K}: {int((n_local == K).sum())}")
