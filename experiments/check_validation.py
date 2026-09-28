"""Проверки валидации (результаты — в eda/05_validation.md).

a) похожа ли валидация на бенчмарк: признаки запросов и доли слоёв;
b) сколько правильных ответов на запрос и у скольких ответов есть история в train;
c) нет ли утечки: строки отложенных запросов не попали в обучающую часть train;
d) санитарные бейзлайны: случайные 50, идеальный ответ, «популярное в городе» — со срезами;
e) стабильность: «популярное в городе» на 5 разных seed валидации и погрешность среднего.

Запуск из корня репозитория (сначала python src/validation.py):
    python experiments/check_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from validation import (DATA, add_flags, add_query_key, build_validation, describe_queries, load_train,
                        load_validation, recall_at_k, region_location_ids, slices)

K = 50
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def popular_in_location(queries: pd.DataFrame, train: pd.DataFrame, corpus_ids: set, k: int = K) -> dict:
    """Топ-k объявлений корпуса по числу выборов в train при поиске в той же search_location_id,
    добор глобальным топом. train — только обучающие строки (train[train_mask])."""
    t = train.loc[train["item_id"].isin(corpus_ids), ["search_location_id", "item_id"]]
    global_top = t["item_id"].value_counts().index[:k].tolist()
    c = (t.groupby(["search_location_id", "item_id"]).size().rename("n").reset_index()
         .sort_values(["search_location_id", "n", "item_id"], ascending=[True, False, True]))
    by_loc = c.groupby("search_location_id").head(k).groupby("search_location_id")["item_id"].agg(list)
    return {qid: list(dict.fromkeys(by_loc.get(loc, []) + global_top))[:k]
            for qid, loc in zip(queries["query_id"], queries["search_location_id"])}


def strata_shares(q: pd.DataFrame) -> pd.Series:
    """Доли 4 слоёв (текст есть в train × есть фильтр), %."""
    return 100 * q[["text_in_train", "has_filter"]].value_counts(normalize=True).sort_index()


def main():
    train = load_train()
    regions = region_location_ids()
    corpus = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])["item_id"].to_numpy()
    corpus_ids = set(corpus)
    val_q, labels, mask = load_validation()
    assert len(mask) == len(train), "train_mask не соответствует train — пересоберите валидацию"

    # a) Бенчмарк: «текст есть в train» — по полному train; валидация: флаги посчитаны по train[mask].
    bench = add_flags(pd.read_parquet(DATA / "benchmark_queries.parquet"), set(train["text_norm"]), regions)
    print("\n=== a) бенчмарк vs валидация ===")
    print(pd.DataFrame({"бенчмарк": describe_queries(bench), "валидация": describe_queries(val_q)}).round(2))
    print("\nдоли слоёв (text_in_train, has_filter), %:")
    print(pd.DataFrame({"бенчмарк": strata_shares(bench), "валидация": strata_shares(val_q)}).round(2))
    # Совпадения точнее текста: у бенчмарка сравниваем с полным train, у валидации — с train[mask].
    tr_fit = train[mask]
    text_loc = lambda d: d["text_norm"] + "|" + d["search_location_id"].astype(str)
    print("\nсовпадает с train, %:")
    print(pd.DataFrame({
        "бенчмарк": {"текст + локация": 100 * text_loc(bench).isin(set(text_loc(train))).mean(),
                     "полный запрос": 100 * add_query_key(bench)["query_key"].isin(set(train["query_key"])).mean()},
        "валидация": {"текст + локация": 100 * text_loc(val_q).isin(set(text_loc(tr_fit))).mean(),
                      "полный запрос": 100 * val_q["query_key"].isin(set(tr_fit["query_key"])).mean()},
    }).round(2))

    # b) Правильные ответы и история.
    n_rel = pd.Series({q: len(v) for q, v in labels.items()})
    print("\n=== b) правильные ответы ===")
    print(f"на запрос: среднее {n_rel.mean():.3f}, максимум {n_rel.max()}, доля с одним {100 * (n_rel == 1).mean():.1f}%")
    print(f"у ответа есть история в обучающем train: {100 * val_q['target_has_history'].mean():.1f}%")
    print(f"все ответы из корпуса: {all(v <= corpus_ids for v in labels.values())}")

    # c) Утечка и согласованность флагов.
    print("\n=== c) утечка ===")
    print(f"строк отложенных запросов в обучающем train: {tr_fit['query_key'].isin(set(val_q['query_key'])).sum()}")
    print(f"отложенных строк: {(~mask).sum()}; строк отложенных запросов в полном train: "
          f"{train['query_key'].isin(set(val_q['query_key'])).sum()}")
    print(f"повторов текста в валидации: {val_q['text_norm'].duplicated().sum()}")
    same = (val_q["text_in_train"] == val_q["text_norm"].isin(set(tr_fit["text_norm"]))).all()
    print(f"флаг text_in_train совпадает с «текст есть в train[mask]»: {same}; "
          f"а по полному train было бы {100 * val_q['text_norm'].isin(set(train['text_norm'])).mean():.0f}% (все)")

    # d) Санитарные бейзлайны.
    rng = np.random.default_rng(0)
    runs = {
        "случайные 50": {q: rng.choice(corpus, K, replace=False).tolist() for q in labels},
        "идеальный ответ": {q: sorted(v) for q, v in labels.items()},
        "популярное в городе": popular_in_location(val_q, tr_fit, corpus_ids),
    }
    print("\n=== d) бейзлайны ===")
    print(f"ожидание для случайных 50: {K / len(corpus):.6f}")
    for name, preds in runs.items():
        print(f"\n{name}:")
        print(slices(recall_at_k(preds, labels, K), val_q).round(4))

    # e) Стабильность по seed валидации.
    print("\n=== e) «популярное в городе» на разных seed ===")
    base_keys = set(val_q["query_key"])
    rows = {}
    for seed in [42, 1, 2, 3, 4]:
        q, lab, m = build_validation(seed=seed, train=train, regions=regions)
        r = recall_at_k(popular_in_location(q, train[m], corpus_ids), lab, K)
        sl = slices(r, q)["recall"]
        rows[seed] = {"recall": r.mean(), "std/sqrt(n)": r.std() / np.sqrt(len(r)),
                      "текста нет в train": sl["текста нет в train"],
                      "ответ без истории": sl["ответ без истории"],
                      "история, %": 100 * q["target_has_history"].mean(),
                      "общих запросов с seed 42, %": 100 * len(base_keys & set(q["query_key"])) / len(q)}
    tab = pd.DataFrame(rows).T
    print(tab.round(4))
    print(f"разброс recall по seed: std {tab['recall'].std():.4f}, "
          f"min–max {tab['recall'].min():.4f}–{tab['recall'].max():.4f}")


if __name__ == "__main__":
    main()
