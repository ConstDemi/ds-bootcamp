"""Бонусы за «качество» объявления поверх готовых кандидатов (eda/15_quality.md).

Гипотеза: среди похожих по тексту объявлений пользователи чаще выбирают «лучшие» — с отзывами,
рейтингом, открытым телефоном, настоящей ценой, подробным описанием. Эти поля есть у всех объявлений
корпуса, поэтому бонус не зависит от истории выборов и одинаково применим к валидации и бенчмарку.

Признаки считаются по корпусу один раз (item_quality_features), затем для каждого запроса переводятся
во внутризапросные перцентили среди его кандидатов (0…1), чтобы масштаб весов был сопоставим между
признаками и запросами. Итоговый скор: score + Σ w_k · f_k, дальше кандидаты пересортировываются.
Новых id не появляется, формат (query_id, item_id, score, rank) сохраняется.

Использование:
    items = pd.read_parquet("data/benchmark_items.parquet", columns=QUALITY_COLS)
    q = item_quality_features(items)
    new = rerank_quality(cands, q, {"pct_rev": 0.002, "phone_hidden": -0.001})
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_COLS = ["item_id", "item_rating", "item_rating_reviews_count", "item_price",
                "item_is_phone_hidden", "item_is_message_forbidden",
                "item_description_raw", "item_title_raw"]

# Признаки, которые переводятся во внутризапросный перцентиль (префикс pct_ в весах)
CONT = ["rev_log", "rating_f", "desc_log", "title_len", "price_log"]
# Бинарные признаки — используются как есть (0/1)
BIN = ["has_rev", "has_rating", "price_stub", "phone_hidden", "msg_forbidden"]


def item_quality_features(items: pd.DataFrame) -> pd.DataFrame:
    """Признаки качества по каждому объявлению корпуса (индекс — item_id).

    rev_log      log1p(число отзывов), NaN → 0
    has_rev      отзывов > 0
    has_rating   рейтинг есть и > 0 (рейтинг 0 — это «нет оценок»)
    rating_f     рейтинг, у кого его нет — 0 (для перцентиля «нет рейтинга» хуже любого)
    rating_c     (рейтинг − 4.5) · [есть рейтинг]
    price_stub   цена ≤ 1 (−1/0/1 — заглушки «договорная»)
    price_log    log1p(цены) для настоящих цен, иначе 0
    phone_hidden, msg_forbidden — флаги из корпуса
    desc_log     log1p(длина описания), title_len — длина заголовка
    """
    rev = items["item_rating_reviews_count"].astype(float).fillna(0)
    rat = items["item_rating"].astype(float)
    price = items["item_price"].astype(float).fillna(-1)
    has_rating = rat.notna() & (rat > 0)
    f = pd.DataFrame({
        "rev_log": np.log1p(rev),
        "has_rev": (rev > 0).astype(np.float32),
        "has_rating": has_rating.astype(np.float32),
        "rating_f": rat.where(has_rating, 0.0),
        "rating_c": ((rat - 4.5) * has_rating).fillna(0.0),
        "price_stub": (price <= 1).astype(np.float32),
        "price_log": np.log1p(price.clip(lower=0)).where(price > 1, 0.0),
        "phone_hidden": items["item_is_phone_hidden"].astype(np.float32),
        "msg_forbidden": items["item_is_message_forbidden"].astype(np.float32),
        "desc_log": np.log1p(items["item_description_raw"].fillna("").str.len()),
        "title_len": items["item_title_raw"].fillna("").str.len().astype(float),
    })
    f.index = items["item_id"].astype(str).to_numpy()
    return f.astype(np.float32)


def candidate_features(cands: pd.DataFrame, qf: pd.DataFrame) -> pd.DataFrame:
    """Кандидаты + признаки качества + внутризапросные перцентили pct_* (0…1, средний ранг при равенствах)."""
    c = cands[["query_id", "item_id", "score", "rank"]].reset_index(drop=True)
    f = qf.reindex(c["item_id"].to_numpy()).reset_index(drop=True).fillna(0)
    c = pd.concat([c, f], axis=1)
    g = c.groupby("query_id", sort=False)
    for col in CONT:
        # перцентиль внутри запроса: у лучшего по признаку ≈1, у худшего ≈0
        c["pct_" + col] = g[col].rank(pct=True, method="average").astype(np.float32)
    return c


def rerank_quality(cands: pd.DataFrame, items_or_qf: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Пересортировка кандидатов по score + Σ w·признак.

    items_or_qf — либо сырые объявления (колонки QUALITY_COLS), либо уже готовый item_quality_features,
    либо уже посчитанные candidate_features (тогда cands игнорируется в части признаков).
    weights — {имя признака: вес}; имена: pct_<CONT>, <BIN>, rating_c, rev_log, desc_log.
    Возвращает query_id, item_id, score, rank в том же формате.
    """
    if all(k in cands.columns for k in weights):
        c = cands
    else:
        qf = item_quality_features(items_or_qf) if "item_id" in items_or_qf.columns else items_or_qf
        c = candidate_features(cands, qf)
    s = c["score"].to_numpy(np.float64).copy()
    for k, w in weights.items():
        if w:
            s += w * c[k].to_numpy(np.float64)
    out = pd.DataFrame({"query_id": c["query_id"].to_numpy(), "item_id": c["item_id"].to_numpy(),
                        "score": s, "old_rank": c["rank"].to_numpy()})
    # сортируем по float64 (при равенстве скора сохраняем прежний порядок), в файл пишем float32
    out = out.sort_values(["query_id", "score", "old_rank"], ascending=[True, False, True], kind="mergesort")
    out["rank"] = out.groupby("query_id", sort=False).cumcount().astype(np.int32) + 1
    out["score"] = out["score"].astype(np.float32)
    return out.drop(columns="old_rank").reset_index(drop=True)
