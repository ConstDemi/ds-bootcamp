"""Sparse-ретривер: BM25 со стеммингом Snowball (русский) по тексту объявления.

Использование:
    idx = SparseIndex.build(items, variant="a")
    cands = idx.search(queries, mode="geo", k=300)   # query_id, item_id, score, rank

queries — таблица в формате benchmark_queries (query_id, search_query, search_location_id, ...),
поэтому один и тот же код гоняет и валидацию, и бенчмарк.

Режимы поиска:
    all — по всему корпусу;
    geo — если search_location_id запроса есть среди item_location_id (город), ищем только
          среди объявлений этого города; регион / «вся Россия» — по всему корпусу. Если в
          городском пуле меньше k объявлений с ненулевым скором, добираем до k лучшими из
          поиска по всему корпусу (без повторов). Поэтому в geo скор может не убывать по rank:
          сначала идут городские, потом добор со своими (глобальными) скорами BM25.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer
from bm25s.tokenization import Tokenized

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text import clean_params, item_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"

_TOKEN_RE = re.compile(r"\w+")
_WS_RE = re.compile(r"\s+")
_STEMMER = Stemmer.Stemmer("russian")  # Snowball; латиницу не трогает


# ---------------------------------------------------------------- тексты документов
def doc_text(items: pd.DataFrame, variant: str = "a") -> pd.Series:
    """Варианты текста документа.

    a — item_text как есть (заголовок + очищенные параметры + 500 символов описания);
    b — то же, но заголовок продублирован (×2 веса заголовка в BM25);
    c — без описания: заголовок + очищенные параметры.
    """
    title = items["item_title_raw"].fillna("").astype(str)
    if variant == "a":
        return item_text(items)
    if variant == "b":
        return title + ". " + item_text(items)
    if variant == "c":
        return title + ". " + clean_params(items["item_infm_params_text"])
    raise ValueError(variant)


# ---------------------------------------------------------------- токенизация
def tokenize(texts, stem_cache: dict | None = None) -> list[list[str]]:
    """Нижний регистр, ё→е, токены \\w+, стемминг Snowball. Стемы кэшируются по словоформе."""
    cache = {} if stem_cache is None else stem_cache
    out = []
    for t in texts:
        words = _TOKEN_RE.findall(str(t).lower().replace("ё", "е"))
        new = [w for w in set(words) if w not in cache]
        if new:
            cache.update(zip(new, _STEMMER.stemWords(new)))
        out.append([cache[w] for w in words])
    return out


def _to_ids(token_lists: list[list[str]]) -> Tokenized:
    """Перевод токенов в id для bm25s (экономит память против списков строк)."""
    vocab: dict[str, int] = {}
    ids = []
    for toks in token_lists:
        ids.append([vocab.setdefault(t, len(vocab)) for t in toks])
    return Tokenized(ids=ids, vocab=vocab)


# ---------------------------------------------------------------- индекс
class SparseIndex:
    """BM25-индекс корпуса (bm25s) на стемах Snowball: поиск по всему корпусу или внутри гео-пула."""
    def __init__(self, retriever: bm25s.BM25, item_ids: np.ndarray, item_loc: np.ndarray, vocab: dict):
        self.bm25 = retriever
        self.item_ids = item_ids          # позиция документа -> item_id (str)
        self.vocab = vocab                # стем -> id столбца
        self.stem_cache: dict = {}
        # позиции объявлений по городу: для режима geo
        order = np.argsort(item_loc, kind="stable")
        locs, starts = np.unique(item_loc[order], return_index=True)
        bounds = np.append(starts, len(order))
        self.city_pos = {int(l): order[bounds[i]:bounds[i + 1]] for i, l in enumerate(locs)}
        self.build_sec = None

    @classmethod
    def build(cls, items: pd.DataFrame, variant: str = "a", k1: float = 1.5, b: float = 0.75):
        """Строит индекс: токенизация и стемминг документов варианта variant, BM25 (k1, b)."""
        t0 = time.time()
        toks = tokenize(doc_text(items, variant))
        tk = _to_ids(toks)
        del toks
        r = bm25s.BM25(k1=k1, b=b)
        r.index(tk, show_progress=False)
        idx = cls(r, items["item_id"].astype(str).to_numpy(), items["item_location_id"].to_numpy(), r.vocab_dict)
        idx.build_sec = time.time() - t0
        return idx

    def scores(self, query: str) -> np.ndarray:
        """BM25-скоры запроса по всем документам. Повторы слов в запросе учитываются один раз."""
        stems = tokenize([query], self.stem_cache)[0]
        cols = sorted({self.vocab[s] for s in stems if s in self.vocab})
        if not cols:
            return np.zeros(len(self.item_ids), dtype=np.float32)
        s = self.bm25.scores
        # scores — CSC-матрица «термин × документ»: суммируем строки терминов запроса
        n = len(self.item_ids)
        out = np.zeros(n, dtype=np.float32)
        for c in cols:
            a, e = s["indptr"][c], s["indptr"][c + 1]
            out[s["indices"][a:e]] += s["data"][a:e]  # в пределах одного термина индексы уникальны
        return out

    @staticmethod
    def _top(pos: np.ndarray, sc: np.ndarray, k: int) -> np.ndarray:
        """Позиции top-k по скору среди pos (только скор > 0), по убыванию."""
        pos = pos[sc[pos] > 0]
        if len(pos) > k:
            pos = pos[np.argpartition(-sc[pos], k - 1)[:k]]
        return pos[np.argsort(-sc[pos], kind="stable")]

    def search(self, queries: pd.DataFrame, mode: str = "all", k: int = 300) -> pd.DataFrame:
        """Топ-k кандидатов на запрос: DataFrame(query_id, item_id, score, rank с 1)."""
        assert mode in ("all", "geo")
        all_pos = np.arange(len(self.item_ids))
        rows_q, rows_p, rows_s = [], [], []
        t0 = time.time()
        for qid, text, loc in zip(queries["query_id"], queries["search_query"], queries["search_location_id"]):
            sc = self.scores(text)
            glob = self._top(all_pos, sc, k)
            if mode == "geo" and int(loc) in self.city_pos:     # город: сначала свой пул
                city = self._top(self.city_pos[int(loc)], sc, k)
                if len(city) < k:                               # добор глобальными без повторов
                    extra = glob[~np.isin(glob, city)][: k - len(city)]
                    city = np.concatenate([city, extra])
                pos = city
            else:
                pos = glob
            rows_q.append(np.full(len(pos), qid, dtype=object))
            rows_p.append(pos)
            rows_s.append(sc[pos])
        self.ms_per_query = 1000 * (time.time() - t0) / max(len(queries), 1)
        pos = np.concatenate(rows_p).astype(np.int64)
        df = pd.DataFrame({"query_id": np.concatenate(rows_q),
                           "item_id": self.item_ids[pos],
                           "score": np.concatenate(rows_s).astype(np.float32)})
        df["rank"] = df.groupby("query_id", sort=False).cumcount().astype(np.int32) + 1
        return df


def to_preds(cands: pd.DataFrame, k: int = 50) -> dict:
    """Кандидаты -> dict query_id -> список item_id (первые k по rank) для recall_at_k."""
    top = cands[cands["rank"] <= k]
    return top.groupby("query_id", sort=False)["item_id"].agg(list).to_dict()


def load_items() -> pd.DataFrame:
    """Колонки корпуса, нужные для текста документа и гео."""
    return pd.read_parquet(DATA / "benchmark_items.parquet",
                           columns=["item_id", "item_title_raw", "item_description_raw",
                                    "item_infm_params_text", "item_location_id"])
