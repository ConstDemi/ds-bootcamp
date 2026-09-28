"""География поиска: карты локаций, города в тексте запроса и поиск по ярусам кандидатов.

Термины:
    город   — search_location_id, который встречается у объявлений (item_location_id);
    регион  — search_location_id, которого нет ни у одного объявления (область, Москва+МО, 621540 = вся Россия).

Что здесь есть:
    load_geo_data()   — корпус и train с координатами и параметрами (для центроидов и словаря городов);
    centroids()       — центроид item_location_id = медиана координат объявлений корпуса + train (это не ответы);
    CityFinder        — поиск названия города в тексте запроса (перенос find_city2 из eda/geo_eda.py);
    GeoMaps           — карты по train (регион → доли городов, пары «search_loc → item_loc»), соседи по центроидам,
                        и правила «запрос → ярусы кандидатов» (RULES);
    tiered_top()      — top-k по ярусам: лучшие из яруса 1, потом яруса 2, …, в конце добор из всего корпуса;
    rrf()             — Reciprocal Rank Fusion позиций двух источников (как fuse_rrf без ярусов в experiments/hybrid.py).

Главное правило против утечки: GeoMaps.build и CityFinder.build получают train, по которому строятся карты
«запрос → выбранное объявление». Для валидации это train[train_mask], для бенчмарка — полный train.
Центроиды и словарь названий строятся по атрибутам объявлений (координаты, адрес) — это не ответы.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
R_EARTH = 6371.0
RUSSIA = 621540          # «вся Россия»


# ---------------------------------------------------------------- данные

def _coords(df: pd.DataFrame) -> pd.DataFrame:
    """Координаты объявлений строками -> float lat/lon."""
    df["lat"] = pd.to_numeric(df.pop("item_latitude"), errors="coerce").astype(float)
    df["lon"] = pd.to_numeric(df.pop("item_longitude"), errors="coerce").astype(float)
    return df


def load_geo_data():
    """(corpus, train): корпус в исходном порядке строк (позиция = индекс в эмбеддингах/BM25)
    и train в исходном порядке (для train_mask). Параметры объявлений train — только уникальные item
    (нужны лишь для словаря названий городов)."""
    icols = ["item_id", "item_location_id", "item_latitude", "item_longitude"]
    corpus = _coords(pd.read_parquet(DATA / "benchmark_items.parquet", columns=icols + ["item_infm_params_text"]))
    corpus["item_id"] = corpus["item_id"].astype(str)
    train = _coords(pd.read_parquet(DATA / "train.parquet",
                                    columns=["search_query", "search_location_id"] + icols))
    train["item_id"] = train["item_id"].astype(str)
    return corpus, train


def train_item_params() -> pd.DataFrame:
    """item_location_id + item_infm_params_text уникальных объявлений train (для словаря городов)."""
    t = pd.read_parquet(DATA / "train.parquet", columns=["item_id", "item_location_id", "item_infm_params_text"])
    return t.drop_duplicates("item_id")[["item_location_id", "item_infm_params_text"]]


def hav(lat1, lon1, lat2, lon2):
    """Расстояние haversine, км (векторно)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def centroids(corpus: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Центроид каждой item_location_id: медиана lat/lon уникальных объявлений корпуса + train."""
    it = pd.concat([corpus[["item_id", "item_location_id", "lat", "lon"]],
                    train[["item_id", "item_location_id", "lat", "lon"]]]).drop_duplicates("item_id")
    return it.groupby("item_location_id").agg(clat=("lat", "median"), clon=("lon", "median"),
                                              n_items=("item_id", "size"))


# ---------------------------------------------------------------- города в тексте (из eda/geo_eda.py)

def norm(s: str) -> str:
    return str(s).lower().replace("ё", "е").strip()


_REG = re.compile(r"(область|край|республик|округ|район|р-н|ао$|автономн)", re.I)
_STREET = re.compile(r"(^ул\.|улица|пр-т|проспект|пер\.|переулок|шоссе|ш\.|бульвар|б-р|наб\.|набережная|"
                     r"площадь|пл\.|мкр|микрорайон|проезд|\d|тупик|аллея|снт|тсн|днт|товарищество|посёлок|поселок|"
                     r"пос\.|жилой комплекс|жк|территория|квартал|станица|село|деревня|тер\.|линия)", re.I)
_PAT = re.compile(r"Место оказания услуг (.+?)(?: Тип | Вид | Как вы | Работаете | Опыт | Где вы | Кто | График |"
                  r" Время | Специальность | Гарантия | Начальная | Куда | Услуга | Стоимость |$)")
ALIAS = {"спб": "санкт-петербург", "питер": "санкт-петербург", "мск": "москва", "екб": "екатеринбург",
         "нск": "новосибирск", "нижний": "нижний новгород"}
BIG = ["москва", "санкт-петербург", "новосибирск", "екатеринбург", "казань", "нижний новгород", "челябинск",
       "самара", "омск", "ростов-на-дону", "уфа", "красноярск", "воронеж", "пермь", "волгоград", "краснодар",
       "саратов", "тюмень", "тольятти", "ижевск", "барнаул", "ульяновск", "иркутск", "хабаровск", "ярославль",
       "владивосток", "махачкала", "томск", "оренбург", "кемерово", "сочи", "тбилиси", "владикавказ", "минск"]
# названия городов, совпадающие с частыми словами услуг
STOP = {"мирный", "белый", "вывоз", "ремонт", "свобода", "мир", "октябрьский", "советский", "заречный",
        "новый", "лесной", "пушкин", "луч", "бор", "радужный", "победа", "северный", "южный", "центральный",
        "восточный", "западный", "кировский", "первомайский", "ленинский", "московский", "раменское",
        "дружба", "звездный", "солнечный", "садовый", "полевой", "зеленый", "степной", "берег", "озерный"}
ENDS = ("", "а", "е", "у", "ом", "ым", "ой", "и", "ы", "ах", "ам", "ского", "ском", "ской", "ске")
_TOK = re.compile(r"[а-яa-z\-]+")


class CityFinder:
    """Название города в тексте запроса -> item_location_id.

    Словарь: для каждой локации самая частая компонента адреса «Место оказания услуг …» (без регионов,
    районов, улиц), + алиасы (спб/мск/питер/екб). Однословные названия ищутся со строгими падежными
    окончаниями, многословные — подстрокой с границами слова. Названия, которые в train не согласуются
    с геопозицией выбранного объявления (ростов(ая кукла), перевоз(ки) …), выбрасываются: это фильтр
    по парам «запрос → объявление», поэтому он строится только по переданному train (без утечки).
    """

    def __init__(self, city2loc: dict, bad: set):
        self.city2loc = city2loc
        self.bad = bad
        names = [n for n in city2loc if n not in STOP]
        multi = [n for n in names if " " in n or "-" in n] + [a for a in BIG if " " in a or "-" in a]
        multi = sorted(set(multi), key=len, reverse=True)
        self.multi_re = [(m, re.compile(r"(?<![а-яa-z\-])" + re.escape(m) + r"(?![а-яa-z])")) for m in multi]
        self.forms = {}
        for name in names:
            if " " in name or "-" in name or len(name) < 4:
                continue
            st = name[:-1] if name[-1] in "аеиоуыяьй" and len(name) >= 5 else name
            for e in ENDS:
                self.forms.setdefault(st + e, name)
            self.forms.setdefault(name, name)

    def _raw(self, query: str):
        """Первый город, упомянутый в тексте запроса (многословные названия раньше однословных), или None."""
        qn = norm(query)
        for m, rx in self.multi_re:
            if rx.search(qn):
                return m
        for t in _TOK.findall(qn):
            if t in ALIAS:
                return ALIAS[t]
            if t in self.forms:
                return self.forms[t]
        return None

    def find(self, query: str):
        """Название города (или None), уже без «негеографичных» названий."""
        n = self._raw(query)
        return None if n is None or n in self.bad else n

    def loc(self, query: str):
        """item_location_id названного города или None."""
        n = self.find(query)
        return None if n is None else self.city2loc.get(n)

    @classmethod
    def build(cls, corpus: pd.DataFrame, train_params: pd.DataFrame, cent: pd.DataFrame,
              train_fit: pd.DataFrame) -> "CityFinder":
        """corpus/train_params — атрибуты объявлений (адреса); train_fit — пары запрос→объявление
        для фильтра «негеографичных» названий (train[train_mask] для валидации, полный train для бенчмарка)."""
        loc_names = defaultdict(Counter)
        allitems = pd.concat([corpus[["item_location_id", "item_infm_params_text"]], train_params])
        for loc, txt in zip(allitems["item_location_id"].to_numpy(), allitems["item_infm_params_text"].to_numpy()):
            m = _PAT.search(txt or "")
            if not m:
                continue
            for part in m.group(1).split(","):
                p = part.strip()
                if not p or _REG.search(p) or _STREET.search(p) or len(p) < 4 or len(p) > 30:
                    continue
                loc_names[loc][norm(p)] += 1
        n_items = cent["n_items"]
        city2loc = {}
        for loc, c in loc_names.items():
            name, k = c.most_common(1)[0]
            if k >= 5 and k >= 0.2 * sum(c.values()):
                # имя уже занято — оставляем локацию с большим числом объявлений
                if name not in city2loc or n_items.get(city2loc[name], 0) < n_items.get(loc, 0):
                    city2loc[name] = loc
        for b in BIG:
            city2loc.setdefault(b, None)
        f = cls(city2loc, set())
        # фильтр «негеографичных» названий по train_fit: выбранное объявление редко рядом с названным городом
        uq = train_fit[["search_query"]].drop_duplicates()
        uq["city"] = uq["search_query"].map(f._raw)
        t = train_fit[["search_query", "lat", "lon"]].merge(uq[uq["city"].notna()], on="search_query")
        cl = t["city"].map(city2loc)
        t["nlat"] = cl.map(cent["clat"]).astype(float)
        t["nlon"] = cl.map(cent["clon"]).astype(float)
        t["dn"] = hav(t["lat"], t["lon"], t["nlat"], t["nlon"])
        cons = t.groupby("city").agg(rows=("dn", "size"), near=("dn", lambda x: (x < 100).mean()))
        f.bad = set(cons[(cons["rows"] >= 5) & (cons["near"] < 0.2)].index)
        return f


# ---------------------------------------------------------------- карты и ярусы

class GeoMaps:
    """Карты локаций и правила «запрос → ярусы кандидатов» (списки позиций корпуса по приоритету).

    Все ярусы одного запроса не пересекаются; после ярусов всегда идёт добор из всего корпуса (в tiered_top).
    """

    def __init__(self, corpus: pd.DataFrame, cent: pd.DataFrame, regions: set, city: CityFinder):
        self.regions = regions
        self.city = city
        loc = corpus["item_location_id"].to_numpy()
        self.item_loc = loc
        order = np.argsort(loc, kind="stable")
        u, st = np.unique(loc[order], return_index=True)
        self.pos = dict(zip(u.tolist(), np.split(order, st[1:])))      # локация -> позиции корпуса
        self.corpus_locs = np.array(sorted(self.pos))
        c = cent.reindex(self.corpus_locs)
        self.cl = np.radians(c[["clat", "clon"]].to_numpy())
        from sklearn.neighbors import BallTree
        ok = ~np.isnan(self.cl).any(1)
        self._tree_locs = self.corpus_locs[ok]
        self._tree = BallTree(self.cl[ok], metric="haversine")
        self.cent = cent
        self._cache: dict = {}

    @classmethod
    def build(cls, corpus, cent, regions, city, train_fit: pd.DataFrame) -> "GeoMaps":
        """Карты по train_fit (только он!): доли городов в регионе и пары «город поиска → город объявления»."""
        m = cls(corpus, cent, regions, city)
        t = train_fit[["search_location_id", "item_location_id"]]
        is_reg = t["search_location_id"].isin(regions)
        r = t[is_reg].groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        r["share"] = r["n"] / r.groupby("search_location_id")["n"].transform("sum")
        r = r.sort_values(["search_location_id", "share"], ascending=[True, False])
        m.region_rows = r.groupby("search_location_id")["n"].sum().to_dict()
        # регион -> DataFrame(item_location_id, share) по убыванию доли; только локации с объявлениями корпуса
        r = r[r["item_location_id"].isin(m.pos)]
        m.region_map = {k: g[["item_location_id", "share"]].reset_index(drop=True)
                        for k, g in r.groupby("search_location_id")}
        # пары «город поиска -> другой город объявления» с долей от всех строк этого города поиска
        c = t[~is_reg].groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        c["share"] = c["n"] / c.groupby("search_location_id")["n"].transform("sum")
        c = c[(c["search_location_id"] != c["item_location_id"]) & c["item_location_id"].isin(m.pos)]
        c = c.sort_values(["search_location_id", "share"], ascending=[True, False])
        m.pairs = c.groupby("search_location_id")["item_location_id"].agg(list).to_dict()
        c1 = c[c["share"] >= 0.01]
        m.pairs1 = c1.groupby("search_location_id")["item_location_id"].agg(list).to_dict()
        return m

    # --- элементарные наборы локаций
    def neighbors(self, loc, radius_km: float) -> list:
        """Локации корпуса в радиусе от центроида loc (без самой loc), по возрастанию расстояния."""
        if loc not in self.cent.index or np.isnan(self.cent.at[loc, "clat"]):
            return []
        p = np.radians(self.cent.loc[[loc], ["clat", "clon"]].to_numpy())
        ind, dist = self.tree_query(p, radius_km)
        return [int(x) for x in self._tree_locs[ind[np.argsort(dist)]] if x != loc]

    def tree_query(self, p, radius_km):
        """Индексы и расстояния (в радианах) локаций в радиусе radius_km от точки p (BallTree, гаверсинус)."""
        ind, dist = self._tree.query_radius(p, r=radius_km / R_EARTH, return_distance=True)
        return ind[0], dist[0]

    def region_locs(self, reg, min_share: float = 0.0) -> list:
        """Города, куда из региона ходят в train, с долей выборов не ниже min_share."""
        m = self.region_map.get(reg)
        if m is None:
            return []
        return m.loc[m["share"] >= min_share, "item_location_id"].tolist()

    def _positions(self, locs) -> np.ndarray:
        """Позиции объявлений корпуса в перечисленных локациях."""
        arr = [self.pos[l] for l in locs if l in self.pos]
        return np.concatenate(arr) if arr else np.empty(0, dtype=np.int64)

    def tiers_from_locs(self, loc_groups: list) -> list:
        """Группы локаций по приоритету -> непересекающиеся ярусы позиций корпуса."""
        seen, out = set(), []
        for g in loc_groups:
            g = [l for l in dict.fromkeys(g) if l not in seen and l in self.pos]
            seen.update(g)
            if g:
                out.append(self._positions(g))
        return out

    # --- правила: (search_loc, названный в тексте город) -> группы локаций
    def loc_groups(self, rule: str, loc, named) -> list:
        """rule — имя правила из RULES. named — item_location_id города из текста или None."""
        is_reg = loc in self.regions
        if rule == "C0R0":          # как сейчас: город -> свой город; регион -> весь корпус
            return [[loc]] if not is_reg else []
        if is_reg:                   # ---- правила для регионов (для городских запросов — как C0)
            full = self.region_locs(loc)
            if rule == "R1":
                return [full]
            if rule.startswith("R2_"):
                return [self.region_locs(loc, float(rule[3:]))]
            if rule == "R3":         # город из текста первым ярусом, потом весь регион
                return ([[named]] if named is not None else []) + [full]
            if rule == "R4a":        # главный город региона, потом остальные
                return [full[:1], full[1:]]
            if rule == "R4b":        # ярусы по доле: ≥10%, 1–10%, <1%
                m = self.region_map.get(loc)
                if m is None:
                    return []
                s = m["share"]
                return [m.loc[s >= .1, "item_location_id"].tolist(),
                        m.loc[(s < .1) & (s >= .01), "item_location_id"].tolist(),
                        m.loc[s < .01, "item_location_id"].tolist()]
            if rule == "R3_4b":      # город из текста, потом ярусы по доле
                return ([[named]] if named is not None else []) + self.loc_groups("R4b", loc, named)
            if rule.startswith("R2t_"):  # ярус 1: города с долей ≥ порога, ярус 2: остальные города региона
                thr = float(rule[4:])
                return [self.region_locs(loc, thr), full]
            if rule.startswith("R32_"):  # город из текста, потом города с долей ≥ порога
                return ([[named]] if named is not None else []) + [self.region_locs(loc, float(rule[4:]))]
            if rule.startswith("RSOFT"):  # плоский пул всего региона; бонус — в bonus_mask
                return [full]
            if rule == "R5":         # только город из текста первым ярусом, иначе весь корпус
                return [[named]] if named is not None else []
            return [[loc]] if not is_reg else []
        # ---- правила для городов (для регионов — как R0)
        if rule.startswith("C1_"):   # свой город -> соседи ≤ R км
            return [[loc], self.neighbors(loc, float(rule[3:]))]
        if rule == "C2":             # свой город -> все пары из train
            return [[loc], self.pairs.get(loc, [])]
        if rule == "C2s":            # свой город -> пары с долей ≥ 1% строк этого города поиска
            return [[loc], self.pairs1.get(loc, [])]
        if rule == "C3":             # свой город -> соседи ≤100 км ∪ все пары
            return [[loc], self.neighbors(loc, 100) + self.pairs.get(loc, [])]
        if rule == "C3s":            # свой город -> соседи ≤100 км ∪ пары ≥ 1%
            return [[loc], self.neighbors(loc, 100) + self.pairs1.get(loc, [])]
        if rule in ("C4", "SOFT"):   # плоский пул без приоритета своего города: свой + соседи ≤100 км + пары ≥ 1%
            return [[loc] + self.neighbors(loc, 100) + self.pairs1.get(loc, [])]
        if rule == "C4all":          # то же со всеми парами
            return [[loc] + self.neighbors(loc, 100) + self.pairs.get(loc, [])]
        return [[loc]]

    def base_rule(self, rule: str) -> str:
        """Мягкие правила берут списки источников из плоского пула: SOFT_w — C4, RSOFT_w — R1."""
        if rule.startswith("SOFT"):
            return "C4"
        if rule.startswith("RSOFT"):
            return "R1"
        return rule

    def bonus_mask(self, rule: str, loc):
        """Для мягких правил: (bool по корпусу, вес бонуса к RRF-скору); иначе (None, 0).
        SOFT_w  — бонус объявлениям своего города;
        RSOFT_w — бонус объявлениям городов региона с долей выборов ≥ 1%."""
        if rule.startswith("SOFT_"):
            return self.item_loc == loc, float(rule.split("_")[1])
        if rule.startswith("RSOFT_"):
            key = ("RSOFT", loc)
            if key not in self._cache:
                self._cache[key] = np.isin(self.item_loc, self.region_locs(loc, 0.01))
            return self._cache[key], float(rule.split("_")[1])
        return None, 0.0

    def tiers(self, rule: str, loc, named=None) -> list:
        """Ярусы гео-пула (позиции объявлений) для правила и локации поиска, с кэшем."""
        key = (rule, loc, named)
        if key not in self._cache:
            self._cache[key] = self.tiers_from_locs(self.loc_groups(rule, loc, named))
        return self._cache[key]


REGION_RULES = ["R1", "R2_0.003", "R2_0.01", "R2_0.02", "R2_0.05", "R2t_0.01", "R32_0.01", "R3", "R4a", "R4b",
                "R3_4b", "R5", "RSOFT_0.005", "RSOFT_0.01", "RSOFT_0.02", "RSOFT_0.03", "RSOFT_0.05"]
CITY_RULES = ["C1_30", "C1_100", "C2", "C2s", "C3", "C3s", "C4", "C4all",
              "SOFT_0.003", "SOFT_0.006", "SOFT_0.01", "SOFT_0.015", "SOFT_0.02", "SOFT_0.03"]
RULES = ["C0R0"] + REGION_RULES + CITY_RULES


# ---------------------------------------------------------------- поиск по ярусам и слияние

def top_in(sc: np.ndarray, pos: np.ndarray, k: int, positive: bool) -> np.ndarray:
    """Позиции top-k по скору среди pos (для BM25 — только скор > 0), по убыванию."""
    if positive:
        pos = pos[sc[pos] > 0]
    if len(pos) > k:
        pos = pos[np.argpartition(-sc[pos], k - 1)[:k]]
    return pos[np.argsort(-sc[pos], kind="stable")]


def tiered_top(sc: np.ndarray, tiers: list, k: int, glob: np.ndarray, positive: bool) -> tuple[np.ndarray, int]:
    """Сначала лучшие из яруса 1, потом яруса 2, …, в конце добор из всего корпуса (glob — глобальный
    топ по убыванию, длиной ≥ k). Возвращает (позиции без повторов, сколько из них взято из ярусов)."""
    out, n = [], 0
    for t in tiers:
        if n >= k:
            break
        p = top_in(sc, t, k - n, positive)
        out.append(p)
        n += len(p)
    n_tier = n
    if n < k:
        picked = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
        extra = glob[~np.isin(glob, picked)][: k - n]
        out.append(extra)
    res = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
    return res.astype(np.int64), n_tier


def rrf(lists: list, k_rrf: int = 60, n_out: int = 300, bonus: np.ndarray | None = None,
        bonus_w: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """RRF по спискам позиций: score = Σ 1/(k_rrf + rank). Порядок при равенстве — как в experiments/hybrid.py
    (первое появление в конкатенации sparse, затем dense). bonus — bool по корпусу (например «свой город»),
    к скору прибавляется bonus_w. Возвращает (позиции, скоры) топ-n_out."""
    cat = np.concatenate(lists)
    s = np.concatenate([1.0 / (k_rrf + np.arange(1, len(l) + 1)) for l in lists])
    u, first, inv = np.unique(cat, return_index=True, return_inverse=True)
    score = np.bincount(inv, weights=s)
    if bonus is not None and bonus_w:
        score = score + bonus_w * bonus[u]
    o = np.lexsort((first, -score))[:n_out]
    return u[o], score[o]
