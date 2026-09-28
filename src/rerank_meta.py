"""Переупорядочивание кандидатов по метаданным: фильтры запроса и микрокатегория (см. eda/11_meta_boost.md).

Вход — кандидаты любого ретривера в общем формате (query_id, item_id str, score, rank с 1),
таблица запросов (query_id, search_query, search_infm_params_text) и таблица объявлений
(item_id, item_infm_params_text, item_microcat_id). Выход — те же кандидаты в том же формате,
но с новым порядком: score = новый скор, rank = 1..n.

Два сигнала:
1. Фильтры. Из строки фильтра запроса берём «Вид услуги X», «Тип услуги X», «Тип услуги автосервиса X»
   (значения «Другое» и пустые игнорируем — они почти ничего не ограничивают). У объявления совпадение
   проверяется подстрокой «<ключ> <значение>» в item_infm_params_text (как в eda/04_text_filters.md).
2. Микрокатегория. Классификатор «текст запроса + фильтр → item_microcat_id» (TF-IDF слова + char 3–5,
   линейная модель). Бонус кандидату = w · P(микрокатегория объявления | запрос).

Базовый скор — по умолчанию 1 / (k + rank): так бонусы не зависят от шкалы score конкретного ретривера
и код работает на любом файле кандидатов (можно взять и исходный score, base="score").

Против утечки: fit_microcat принимает train-таблицу. Для валидации передавайте train[train_mask],
для бенчмарка — полный train.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ фильтры

# Словарь ключей фильтра — тот же, что в eda/text_eda.py (там скрипт, импортировать нельзя).
FILTER_KEYS = ["Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Онлайн-запись", "Онлайн-бронирование",
               "Кто оказывает услуги", "Срочная услуга (мультистатус)", "Рейтинг пользователя", "Поиск по слотам",
               "Сортировка для URL", "Где вы оказываете услуги", "Предмет или специальность", "Чем вы занимаетесь",
               "Ваши клиенты", "Специальность или сфера", "Слова в описании", "Услуги", "Услуга",
               "Участие в пилоте CPL", "Аренда авто", "Гарантия", "Коробка передач", "Класс авто"]
_KEY_RE = re.compile("(" + "|".join(re.escape(k) for k in sorted(FILTER_KEYS, key=len, reverse=True)) + r")(?=\s|$)")
CORE_KEYS = ("Вид услуги", "Тип услуги", "Тип услуги автосервиса")
IGNORE_VALUES = {"", "Другое"}


def parse_filters(s: str) -> list[tuple[str, str]]:
    """Строка фильтра -> список пар (ключ, значение). Копия parse_filters из eda/text_eda.py."""
    if not isinstance(s, str) or not s.strip():
        return []
    ms = list(_KEY_RE.finditer(s))
    if not ms:
        return [("<other>", s.strip())]
    parts = []
    if ms[0].start() > 0:
        parts.append(("<other>", s[: ms[0].start()].strip()))
    # слово-ключ сразу после «Вид/Тип услуги» на самом деле значение («Тип услуги Аренда авто») — пропускаем
    keep = []
    for m in ms:
        if keep and keep[-1].group(1) in CORE_KEYS and not s[keep[-1].end():m.start()].strip() \
                and m.group(1) in ("Услуги", "Услуга", "Аренда авто"):
            continue
        keep.append(m)
    for j, m in enumerate(keep):
        end = keep[j + 1].start() if j + 1 < len(keep) else len(s)
        parts.append((m.group(1), s[m.end():end].strip()))
    return parts


def core_filters(s: str) -> dict[str, list[str]]:
    """Значимые пары фильтра: {"vid": [...], "tip": [...]} в виде готовых подстрок «<ключ> <значение>».
    «Тип услуги автосервиса» относим к «tip». «Другое» и пустые значения выбрасываем.
    Несколько значений одного ключа трактуются как ИЛИ."""
    out = {"vid": [], "tip": []}
    for k, v in parse_filters(s):
        if k not in CORE_KEYS or v in IGNORE_VALUES:
            continue
        out["vid" if k == "Вид услуги" else "tip"].append(f"{k} {v}")
    return out


def filter_match(cands: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к кандидатам колонки:
    q_vid / q_tip  — у запроса есть значимый Вид / Тип;
    m_vid / m_tip  — объявление им соответствует (False, если у запроса такого ключа нет);
    m_all          — объявление соответствует всем значимым ключам запроса (у запроса есть хоть один)."""
    qf = {qid: core_filters(s) for qid, s in
          zip(queries["query_id"], queries["search_infm_params_text"].fillna(""))}
    params = items.set_index("item_id")["item_infm_params_text"].fillna("")
    c = cands.copy()
    ip = params.reindex(c["item_id"]).to_numpy()
    n = len(c)
    q_vid, q_tip = np.zeros(n, bool), np.zeros(n, bool)
    m_vid, m_tip = np.zeros(n, bool), np.zeros(n, bool)
    for i, (qid, p) in enumerate(zip(c["query_id"].to_numpy(), ip)):
        f = qf.get(qid)
        if not f or not (f["vid"] or f["tip"]):
            continue
        p = p if isinstance(p, str) else ""
        if f["vid"]:
            q_vid[i] = True
            m_vid[i] = any(v in p for v in f["vid"])
        if f["tip"]:
            q_tip[i] = True
            m_tip[i] = any(v in p for v in f["tip"])
    c["q_vid"], c["q_tip"], c["m_vid"], c["m_tip"] = q_vid, q_tip, m_vid, m_tip
    c["m_all"] = (q_vid | q_tip) & (m_vid | ~q_vid) & (m_tip | ~q_tip)
    return c


# ------------------------------------------------------------------ классификатор микрокатегорий

def query_doc(text: pd.Series, filt: pd.Series) -> pd.Series:
    """Текст для классификатора: нормализованный запрос + значимые значения фильтра (Вид/Тип)."""
    t = text.fillna("").astype(str).str.lower().str.replace("ё", "е", regex=False)
    f = filt.fillna("").map(lambda s: " ".join(sum(core_filters(s).values(), [])))
    return (t + " | " + f.str.lower().str.replace("ё", "е", regex=False)).str.strip(" |")


class MicrocatModel:
    """TF-IDF (слова 1–2 + char_wb 3–5) + SGD-логрегрессия (one-vs-rest), вероятности нормированы."""

    def __init__(self, vec_w, vec_c, clf):
        self.vec_w, self.vec_c, self.clf = vec_w, vec_c, clf
        self.classes_ = None if clf is None else clf.classes_

    def _x(self, docs):
        from scipy.sparse import hstack
        return hstack([self.vec_w.transform(docs), self.vec_c.transform(docs)]).tocsr()

    def predict_proba(self, queries: pd.DataFrame) -> np.ndarray:
        """Вероятности микрокатегорий для запросов (текст + фильтры), нормированные на 1."""
        docs = query_doc(queries["search_query"], queries["search_infm_params_text"])
        p = self.clf.predict_proba(self._x(docs))
        return p / np.maximum(p.sum(1, keepdims=True), 1e-12)


def fit_microcat(train: pd.DataFrame, weighting: str = "text", max_iter: int = 8, alpha: float = 2e-6, seed: int = 0) -> MicrocatModel:
    """Учит классификатор «запрос (+ фильтр) → item_microcat_id».

    train — строки train с колонками search_query, search_infm_params_text, item_microcat_id.
    ДЛЯ ВАЛИДАЦИИ — только train[train_mask], для бенчмарка — полный train.
    Строки схлопываются в уникальные (документ, микрокатегория) с весом:
      weighting="text" — каждый уникальный текст весит 1 (как в бенчмарке: один запрос на текст),
                          внутри текста вес делится пропорционально числу выборов;
      weighting="rows" — вес = число строк (частые тексты доминируют).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier

    d = pd.DataFrame({"doc": query_doc(train["search_query"], train["search_infm_params_text"]),
                      "y": train["item_microcat_id"].to_numpy()})
    g = d.groupby(["doc", "y"]).size().rename("n").reset_index()
    if weighting == "text":
        g["w"] = g["n"] / g.groupby("doc")["n"].transform("sum")
    else:
        g["w"] = g["n"].astype(float)
    vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True,
                            token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=100_000,
                            sublinear_tf=True, dtype=np.float32)
    vec_w.fit(g["doc"].drop_duplicates())
    vec_c.fit(g["doc"].drop_duplicates())
    m = MicrocatModel(vec_w, vec_c, None)
    x = m._x(g["doc"])
    clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=max_iter, tol=None, n_jobs=-1, random_state=seed)
    clf.fit(x, g["y"].to_numpy(), sample_weight=g["w"].to_numpy())
    m.clf, m.classes_ = clf, clf.classes_
    return m


def microcat_prob(cands: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame,
                  model: MicrocatModel | None = None, proba: np.ndarray | None = None) -> pd.DataFrame:
    """Добавляет колонку p_mc = P(микрокатегория объявления | запрос) и mc_rank (1 = топ-1 предсказание).
    proba можно передать заранее посчитанной (строки в порядке queries)."""
    if proba is None:
        proba = model.predict_proba(queries)
    classes = model.classes_ if model is not None else None
    cls_idx = {c: i for i, c in enumerate(classes)}
    q_idx = {q: i for i, q in enumerate(queries["query_id"])}
    mc = items.set_index("item_id")["item_microcat_id"]
    c = cands.copy()
    qi = c["query_id"].map(q_idx).to_numpy()
    ci = c["item_id"].map(mc).map(cls_idx).fillna(-1).astype(int).to_numpy()
    ok = ci >= 0
    p = np.zeros(len(c), np.float32)
    p[ok] = proba[qi[ok], ci[ok]]
    # ранг микрокатегории объявления среди предсказаний запроса
    order = np.argsort(-proba, axis=1)
    rank_of = np.empty_like(order)
    rank_of[np.arange(len(order))[:, None], order] = np.arange(proba.shape[1])[None, :] + 1
    r = np.full(len(c), 10_000, np.int32)
    r[ok] = rank_of[qi[ok], ci[ok]]
    c["p_mc"], c["mc_rank"] = p, r
    return c


def prf_microcat(cands: pd.DataFrame, items: pd.DataFrame, top: int = 20) -> pd.Series:
    """Без обучения: доля микрокатегории объявления среди топ-`top` кандидатов того же запроса
    (псевдо-релевантная обратная связь). Возвращает Series в порядке cands."""
    mc = items.set_index("item_id")["item_microcat_id"]
    c = cands[["query_id", "rank"]].copy()
    c["mc"] = cands["item_id"].map(mc).to_numpy()
    head = c[c["rank"] <= top]
    share = head.groupby(["query_id", "mc"]).size() / head.groupby("query_id").size()
    return pd.Series(share.reindex(pd.MultiIndex.from_frame(c[["query_id", "mc"]])).fillna(0).to_numpy(),
                     index=cands.index)


# ------------------------------------------------------------------ переупорядочивание

def base_score(cands: pd.DataFrame, base: str = "rrf", k: int = 60) -> np.ndarray:
    """Базовый скор: "rrf" = 1/(k + rank) (не зависит от шкалы), "score" = исходный score."""
    if base == "score":
        return cands["score"].to_numpy(np.float64)
    return 1.0 / (k + cands["rank"].to_numpy(np.float64))


def _finish(c: pd.DataFrame, new_score: np.ndarray) -> pd.DataFrame:
    """Сортирует по новому скору внутри запроса (при равенстве — по старому рангу), пересчитывает rank."""
    out = pd.DataFrame({"query_id": c["query_id"].to_numpy(), "item_id": c["item_id"].to_numpy(),
                        "score": new_score.astype(np.float32), "old_rank": c["rank"].to_numpy()})
    out = out.sort_values(["query_id", "score", "old_rank"], ascending=[True, False, True], kind="mergesort")
    out["rank"] = out.groupby("query_id", sort=False).cumcount().astype(np.int32) + 1
    return out[["query_id", "item_id", "score", "rank"]].reset_index(drop=True)


def rerank_bonus(feats: pd.DataFrame, w_vid: float = 0.0, w_tip: float = 0.0, w_mc: float = 0.0,
                 w_prf: float = 0.0, base: str = "rrf", k: int = 60, mc_mode: str = "prob",
                 mc_topk: int = 3) -> pd.DataFrame:
    """Аддитивный бонус к базовому скору.
    feats — кандидаты после filter_match (и microcat_prob / колонки prf, если нужны).
    Шкала бонуса: при base="rrf", k=60 бонус 0.005 ≈ подъём с 50-го места (1/110) примерно на 25-е.
    mc_mode: "prob" — w_mc · P(mc | q); "topk" — w_mc, если микрокатегория объявления в top-k предсказаний."""
    s = base_score(feats, base, k)
    if w_vid:
        s = s + w_vid * feats["m_vid"].to_numpy()
    if w_tip:
        s = s + w_tip * feats["m_tip"].to_numpy()
    if w_mc:
        if mc_mode == "prob":
            s = s + w_mc * feats["p_mc"].to_numpy()
        else:
            s = s + w_mc * (feats["mc_rank"].to_numpy() <= mc_topk)
    if w_prf:
        s = s + w_prf * feats["prf"].to_numpy()
    return _finish(feats, s)


def rerank_quota(feats: pd.DataFrame, reserve: int = 5, top_k: int = 50, match_col: str = "m_all",
                 inner: pd.DataFrame | None = None) -> pd.DataFrame:
    """«Жёстко с запасом»: в первых top_k местах — лучшие (top_k − reserve) совпавших с фильтром,
    затем `reserve` лучших несовпавших; если совпавших мало, места добирают несовпавшие.
    Запросы без значимого фильтра не меняются. Хвост (места > top_k) — остальные в исходном порядке.
    inner — порядок внутри групп (например, уже переупорядоченный бонусом по микрокатегории);
    по умолчанию исходный rank."""
    c = feats if inner is None else inner.merge(feats[["query_id", "item_id", match_col, "q_vid", "q_tip"]],
                                                on=["query_id", "item_id"], how="left")
    c = c.sort_values(["query_id", "rank"], kind="mergesort")
    has_f = (c["q_vid"] | c["q_tip"]).to_numpy()
    match = c[match_col].to_numpy()
    new_rank = c["rank"].to_numpy().astype(np.int64).copy()
    qids = c["query_id"].to_numpy()
    # границы запросов
    starts = np.flatnonzero(np.r_[True, qids[1:] != qids[:-1]])
    ends = np.r_[starts[1:], len(qids)]
    for a, b in zip(starts, ends):
        if not has_f[a]:
            continue
        idx = np.arange(a, b)
        m, nm = idx[match[a:b]], idx[~match[a:b]]
        n_m = min(len(m), top_k - min(reserve, len(nm)))
        n_nm = min(top_k - n_m, len(nm))
        head = np.r_[m[:n_m], nm[:n_nm]]
        tail_mask = np.ones(b - a, bool)
        tail_mask[head - a] = False
        tail = idx[tail_mask]
        order = np.r_[head, tail]
        new_rank[order] = np.arange(1, b - a + 1)
    out = pd.DataFrame({"query_id": qids, "item_id": c["item_id"].to_numpy(), "rank": new_rank})
    out["score"] = (1.0 / (60 + out["rank"])).astype(np.float32)
    out = out.sort_values(["query_id", "rank"], kind="mergesort").reset_index(drop=True)
    out["rank"] = out["rank"].astype(np.int32)
    return out[["query_id", "item_id", "score", "rank"]]


def load_items(path, columns=("item_id", "item_infm_params_text", "item_microcat_id")) -> pd.DataFrame:
    """Таблица объявлений корпуса с нужными колонками."""
    return pd.read_parquet(path, columns=list(columns))
