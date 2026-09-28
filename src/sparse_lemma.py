"""BM25 с лемматизацией (pymorphy3) вместо стемминга или вместе с ним.

Всё, кроме токенизации, берётся из src/sparse.py без изменений: текст документа (doc_text),
параметры BM25, поиск в режимах all/geo и добор (SparseIndex.search). Подменяется только то,
как слово превращается в термины индекса:

    stem        — стем Snowball, как в sparse.py (контрольный вариант);
    lemma       — нормальная форма pymorphy3: parse(word)[0].normal_form;
    lemma+stem  — каждое слово даёт два термина: "L:<лемма>" и "S:<стем>". Префиксы не дают
                  лемме и стему слиться в один термин, если они совпали по написанию.

Нормализация одинаковая для всех вариантов: нижний регистр, ё→е, токены \\w+.
Лемму тоже приводим ё→е: pymorphy иногда возвращает нормальную форму с «ё».

Использование:
    idx = LemmaIndex.build(items, scheme="lemma", variant="b")
    cands = idx.search(queries, mode="geo", k=300)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import pymorphy3
from bm25s.tokenization import Tokenized

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparse import _STEMMER, _TOKEN_RE, SparseIndex, doc_text  # noqa: E402

SCHEMES = ("stem", "lemma", "lemma+stem")
_MORPH = None


def morph() -> pymorphy3.MorphAnalyzer:
    """Анализатор создаётся один раз (загрузка словарей ~1 с)."""
    global _MORPH
    if _MORPH is None:
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


class TermMapper:
    """Слово -> кортеж id терминов. Кэш по уникальной словоформе: лемматизация дорогая,
    а уникальных слов в корпусе на два порядка меньше, чем словоупотреблений."""

    def __init__(self, scheme: str):
        assert scheme in SCHEMES, scheme
        self.scheme = scheme
        self.vocab: dict[str, int] = {}          # термин -> id столбца
        self.cache: dict[str, tuple] = {}        # словоформа -> id терминов
        self.lemma_sec = 0.0                     # время на pymorphy + Snowball

    def terms(self, words: list[str]) -> list[list[str]]:
        """Термины для списка новых словоформ (без кэша)."""
        t0 = time.time()
        out = []
        if self.scheme in ("stem", "lemma+stem"):
            stems = _STEMMER.stemWords(words)
        if self.scheme in ("lemma", "lemma+stem"):
            m = morph()
            lemmas = [m.parse(w)[0].normal_form.replace("ё", "е") for w in words]
        for i in range(len(words)):
            if self.scheme == "stem":
                out.append([stems[i]])
            elif self.scheme == "lemma":
                out.append([lemmas[i]])
            else:
                out.append(["L:" + lemmas[i], "S:" + stems[i]])
        self.lemma_sec += time.time() - t0
        return out

    def encode(self, texts, grow: bool = True) -> list[list[int]]:
        """Тексты -> списки id терминов. grow=False (запросы): неизвестные термины отбрасываем."""
        out = []
        for t in texts:
            words = _TOKEN_RE.findall(str(t).lower().replace("ё", "е"))
            new = [w for w in dict.fromkeys(words) if w not in self.cache]
            if new:
                for w, ts in zip(new, self.terms(new)):
                    if grow:
                        ids = tuple(self.vocab.setdefault(x, len(self.vocab)) for x in ts)
                    else:
                        ids = tuple(self.vocab[x] for x in ts if x in self.vocab)
                    self.cache[w] = ids
            out.append([i for w in words for i in self.cache[w]])
        return out

    def word_terms(self, word: str) -> list[str]:
        """Термины одного слова строками — для разбора примеров."""
        return self.terms([word])[0]


class LemmaIndex(SparseIndex):
    """SparseIndex с подменённой токенизацией; поиск (search, _top) унаследован."""

    @classmethod
    def build(cls, items: pd.DataFrame, scheme: str = "lemma", variant: str = "b",
              k1: float = 1.5, b: float = 0.75):
        """Строит индекс: каждое слово -> термины леммы и/или стема (scheme), BM25 (k1, b)."""
        t0 = time.time()
        mapper = TermMapper(scheme)
        ids = mapper.encode(doc_text(items, variant))
        tk = Tokenized(ids=ids, vocab=mapper.vocab)
        del ids
        r = bm25s.BM25(k1=k1, b=b)
        r.index(tk, show_progress=False)
        idx = cls(r, items["item_id"].astype(str).to_numpy(), items["item_location_id"].to_numpy(), r.vocab_dict)
        # после индексации словарь больше не растёт: новые слова запросов в индекс не попадут
        idx.mapper = mapper
        idx.build_sec = time.time() - t0
        idx.n_words = len(mapper.cache)
        return idx

    def query_cols(self, query: str) -> list[int]:
        """id столбцов матрицы для терминов запроса (повторы — один раз)."""
        # кэш слов общий с документами, поэтому слово запроса, которого не было в корпусе,
        # кэшируется с пустым набором терминов (grow=False)
        return sorted(set(self.mapper.encode([query], grow=False)[0]))

    def scores(self, query: str) -> np.ndarray:
        """BM25-скоры запроса по всем документам (как SparseIndex.scores, но свои термины)."""
        n = len(self.item_ids)
        out = np.zeros(n, dtype=np.float32)
        s = self.bm25.scores
        for c in self.query_cols(query):
            a, e = s["indptr"][c], s["indptr"][c + 1]
            out[s["indices"][a:e]] += s["data"][a:e]
        return out
