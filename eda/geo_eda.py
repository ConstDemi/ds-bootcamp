"""EDA: география (location_id, координаты, города в тексте запроса)."""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

D = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data") + "/"   # data/ в корне репозитория
R_EARTH = 6371.0
pd.set_option("display.width", 200)


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def q(s, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    s = pd.Series(s).dropna()
    return {f"p{int(x*100)}": round(float(s.quantile(x)), 1) for x in qs} | {"n": len(s)}


icols = ["item_id", "item_location_id", "item_latitude", "item_longitude", "item_infm_params_text"]
tr = pd.read_parquet(D + "train.parquet", columns=["search_query", "search_location_id",
                                                  "search_is_delivery_search"] + icols)
co = pd.read_parquet(D + "benchmark_items.parquet", columns=icols)
bq = pd.read_parquet(D + "benchmark_queries.parquet",
                     columns=["query_id", "search_query", "search_location_id", "search_is_delivery_search"])
for df in (tr, co):
    df["lat"] = pd.to_numeric(df.item_latitude, errors="coerce").astype(float)
    df["lon"] = pd.to_numeric(df.item_longitude, errors="coerce").astype(float)
    df.drop(columns=["item_latitude", "item_longitude"], inplace=True)

print("NaN coords train/corpus:", tr.lat.isna().mean(), co.lat.isna().mean())
print("delivery train:", tr.search_is_delivery_search.value_counts().to_dict(),
      "bench:", bq.search_is_delivery_search.value_counts().to_dict())

# ---------- центроиды location_id ----------
items = pd.concat([co[["item_id", "item_location_id", "lat", "lon"]],
                   tr[["item_id", "item_location_id", "lat", "lon"]]]).drop_duplicates("item_id")
print("unique items (corpus+train):", len(items), "overlap train items in corpus:",
      tr.item_id.drop_duplicates().isin(co.item_id).mean())
cent = items.groupby("item_location_id").agg(clat=("lat", "median"), clon=("lon", "median"),
                                             n_items=("item_id", "size"))
items = items.join(cent, on="item_location_id")
items["d_c"] = hav(items.lat, items.lon, items.clat, items.clon)
rad = items.groupby("item_location_id").d_c.agg(r50="median", r90=lambda x: x.quantile(0.9))
cent = cent.join(rad)
print("\n# locations:", len(cent))
big = cent[cent.n_items >= 30]
print("radius per location (median over locs with >=30 items), r50:", q(big.r50), "\n r90:", q(big.r90))
print("items-weighted r50 distribution:", q(np.repeat(cent.r50.values, cent.n_items.values)))
print("top locations by items:\n", cent.sort_values("n_items", ascending=False).head(12))

# ---------- Q1 ----------
tr["match"] = tr.search_location_id == tr.item_location_id
print("\n## Q1 match share overall:", tr.match.mean())
print(tr.groupby("search_is_delivery_search").match.agg(["mean", "size"]))
tr = tr.join(cent[["clat", "clon", "r50"]].rename(columns=lambda c: "s_" + c), on="search_location_id")
tr["dist"] = hav(tr.lat, tr.lon, tr.s_clat, tr.s_clon)
mm = tr[~tr.match]
print("mismatch rows:", len(mm), "search loc w/o centroid:", mm.s_clat.isna().mean())
print("dist (km) mismatched:", q(mm.dist))
print("dist (km) matched (sanity):", q(tr[tr.match].dist))
# search locations that never appear as item location
s_only = set(tr.search_location_id) - set(items.item_location_id)
print("search locs not in item locs:", len(s_only), "rows:", tr.search_location_id.isin(s_only).mean())
# top mismatching pairs
top_pairs = mm.groupby(["search_location_id", "item_location_id"]).size().sort_values(ascending=False).head(10)
print("top mismatch pairs:\n", top_pairs)
# locations which are mostly 'wider' (search loc that fan out to many item locs)
fan = tr.groupby("search_location_id").agg(n=("match", "size"), match=("match", "mean"),
                                            n_item_locs=("item_location_id", "nunique"))
print("search locs with >=200 rows, lowest match share:\n",
      fan[fan.n >= 200].sort_values("match").head(10).join(cent[["n_items", "r50", "r90"]]))
print("share of rows from search locs with match<0.5:",
      fan.loc[fan.match < 0.5, "n"].sum() / len(tr))

# ---------- Q2 ----------
corpus_locs = set(co.item_location_id)
bq["in_corpus"] = bq.search_location_id.isin(corpus_locs)
print("\n## Q2 bench loc in corpus locs:", bq.in_corpus.mean(),
      "in train search locs:", bq.search_location_id.isin(set(tr.search_location_id)).mean())
cnt = co.item_location_id.value_counts()
bq["n_exact"] = bq.search_location_id.map(cnt).fillna(0).astype(int)
print("corpus items in exact loc:", q(bq.n_exact, (0, .1, .25, .5, .75, .9, 1)))
print("share queries with n_exact<50:", (bq.n_exact < 50).mean(), "==0:", (bq.n_exact == 0).mean())
cc = co.dropna(subset=["lat"])
tree = BallTree(np.radians(cc[["lat", "lon"]].values), metric="haversine")
bq = bq.join(cent[["clat", "clon"]], on="search_location_id")
has = bq.clat.notna()
print("bench queries with centroid:", has.mean())
X = np.radians(bq.loc[has, ["clat", "clon"]].values)
for R in (30, 100, 300):
    bq.loc[has, f"n_R{R}"] = tree.query_radius(X, r=R / R_EARTH, count_only=True)
    print(f"corpus items within {R} km:", q(bq[f"n_R{R}"], (0, .1, .25, .5, .75, .9, 1)))
print("corpus size:", len(co))

# ---------- Q3 ----------
print("\n## Q3 loss if hard filter")
res = []
for dflag, g in tr.groupby("search_is_delivery_search"):
    row = {"delivery": dflag, "n": len(g), "exact_loss": 1 - g.match.mean()}
    for R in (30, 100, 300):
        keep = g.match | (g.dist <= R)  # объединение: exact OR в радиусе
        row[f"R{R}_loss"] = 1 - (g.dist <= R).mean()
        row[f"exact_or_R{R}_loss"] = 1 - keep.mean()
    res.append(row)
g = tr
row = {"delivery": "all", "n": len(g), "exact_loss": 1 - g.match.mean()}
for R in (30, 100, 300):
    row[f"R{R}_loss"] = 1 - (g.dist <= R).mean()
    row[f"exact_or_R{R}_loss"] = 1 - (g.match | (g.dist <= R)).mean()
res.append(row)
print(pd.DataFrame(res).round(4).to_string())
# per-unique (query, loc) — macro-ish: share of query groups where ANY chosen item is out of loc
grp = tr.groupby(["search_query", "search_location_id"]).match.agg(["mean", "size"])
print("per (query,loc) group: mean match share (macro):", grp["mean"].mean(),
      "share groups with all matched:", (grp["mean"] == 1).mean())
# how far do mismatches go relative to location radius
print("mismatch dist / search-loc r50 ratio:", q(mm.dist / mm.s_r50.replace(0, np.nan)))

# ---------- Q4: города в тексте ----------
print("\n## Q4 city names")
REG = re.compile(r"(область|край|республик|округ|район|р-н|ао$|автономн)", re.I)
STREET = re.compile(r"(^ул\.|улица|пр-т|проспект|пер\.|переулок|шоссе|ш\.|бульвар|б-р|наб\.|набережная|"
                    r"площадь|пл\.|мкр|микрорайон|проезд|\d|тупик|аллея|снт|тсн|днт|товарищество|посёлок|поселок|"
                    r"пос\.|жилой комплекс|жк|территория|квартал|станица|село|деревня|тер\.|линия)", re.I)
PAT = re.compile(r"Место оказания услуг (.+?)(?: Тип | Вид | Как вы | Работаете | Опыт | Где вы | Кто | График |"
                 r" Время | Специальность | Гарантия | Начальная | Куда | Услуга | Стоимость |$)")


def norm(s):
    return s.lower().replace("ё", "е").strip()


loc_names = defaultdict(Counter)
allitems = pd.concat([co[["item_location_id", "item_infm_params_text"]],
                      tr[["item_location_id", "item_infm_params_text"]].drop_duplicates()])
for loc, txt in zip(allitems.item_location_id.values, allitems.item_infm_params_text.values):
    m = PAT.search(txt or "")
    if not m:
        continue
    for part in m.group(1).split(","):
        p = part.strip()
        if not p or REG.search(p) or STREET.search(p) or len(p) < 4 or len(p) > 30:
            continue
        loc_names[loc][norm(p)] += 1
del allitems
city2loc = {}
for loc, c in loc_names.items():
    tot = cent.n_items.get(loc, 0)
    name, k = c.most_common(1)[0]
    if k >= 5 and k >= 0.2 * sum(c.values()):
        # если имя уже занято — оставить локацию с большим числом айтемов
        if name not in city2loc or cent.n_items.get(city2loc[name], 0) < tot:
            city2loc[name] = loc
print("city vocab size:", len(city2loc), "sample:", list(city2loc)[:40])
# ручные алиасы крупных городов
alias = {"спб": "санкт-петербург", "питер": "санкт-петербург", "мск": "москва", "екб": "екатеринбург",
         "нск": "новосибирск", "нижний": "нижний новгород"}
BIG = ["москва", "санкт-петербург", "новосибирск", "екатеринбург", "казань", "нижний новгород", "челябинск",
       "самара", "омск", "ростов-на-дону", "уфа", "красноярск", "воронеж", "пермь", "волгоград", "краснодар",
       "саратов", "тюмень", "тольятти", "ижевск", "барнаул", "ульяновск", "иркутск", "хабаровск", "ярославль",
       "владивосток", "махачкала", "томск", "оренбург", "кемерово", "сочи", "тбилиси", "владикавказ", "минск"]
for b in BIG:
    city2loc.setdefault(b, None)
# стоп-слова: имена городов, совпадающие с частыми словами услуг
STOP = {"мирный", "белый", "вывоз", "ремонт", "свобода", "мир", "октябрьский", "советский", "заречный",
        "новый", "лесной", "пушкин", "луч", "бор", "радужный", "победа", "северный", "южный", "центральный",
        "восточный", "западный", "кировский", "первомайский", "ленинский", "московский", "раменское",
        "дружба", "звездный", "солнечный", "садовый", "полевой", "зеленый", "степной", "берег", "озерный"}
single = {}  # stem -> name
multi = []
for name in city2loc:
    if name in STOP:
        continue
    if " " in name or "-" in name:
        multi.append(name)
    else:
        if len(name) >= 5:
            single[name[:-1] if name[-1] in "аеиоуыяьй" else name] = name
        elif len(name) >= 4:
            single[name] = name
stems_by_len = sorted(single, key=len, reverse=True)
single_set = set(single)
TOK = re.compile(r"[а-яa-z\-]+")


def find_city(query):
    qn = norm(query)
    for m in multi:
        if m in qn:
            return m
    for t in TOK.findall(qn):
        if t in alias:
            return alias[t]
        # проверяем префиксы токена как стемы (падежные окончания до 3 символов)
        for cut in range(0, 4):
            st = t[: len(t) - cut] if cut else t
            if len(st) >= 4 and st in single_set:
                return single[st]
    return None


bq["city"] = bq.search_query.map(find_city)
uq = tr[["search_query"]].drop_duplicates()
uq["city"] = uq.search_query.map(find_city)
tr = tr.merge(uq, on="search_query", how="left")
print("bench share with city:", bq.city.notna().mean(), "train rows:", tr.city.notna().mean(),
      "train uniq queries:", uq.city.notna().mean())
print("bench top cities:", bq.city.value_counts().head(25).to_dict())
print("bench examples:", bq.loc[bq.city.notna(), ["search_query", "city"]].sample(25, random_state=0).values.tolist())
bigset = set(BIG) | set(alias.values())
bq["city_big"] = bq.city.isin(bigset)
tr["city_big"] = tr.city.isin(bigset)
print("big-city-only share bench:", bq.city_big.mean(), "train:", tr.city_big.mean())
print("\nmatch share: with city", tr[tr.city.notna()].match.mean(), " without", tr[tr.city.isna()].match.mean())
print("match share: big city mention", tr[tr.city_big].match.mean())
tr["city_loc"] = tr.city.map(city2loc)
wc = tr[tr.city_loc.notna()]
print("rows with city mapped to loc:", len(wc))
print(" city_loc==search_loc:", (wc.city_loc == wc.search_location_id).mean(),
      " city_loc==item_loc:", (wc.city_loc == wc.item_location_id).mean())
wd = wc[wc.city_loc != wc.search_location_id]
print(" among city!=search loc (n=%d): item_loc==city_loc %.3f, item_loc==search_loc %.3f" %
      (len(wd), (wd.city_loc == wd.item_location_id).mean(), wd.match.mean()))
print(" dist to search centroid for those:", q(wd.dist))
bq["city_loc"] = bq.city.map(city2loc)
print("bench: city mapped", bq.city_loc.notna().mean(), " city_loc != search loc:",
      (bq.city_loc.notna() & (bq.city_loc != bq.search_location_id)).mean())
print(tr[tr.city.notna() & ~tr.match].sample(20, random_state=1)[
    ["search_query", "city", "search_location_id", "item_location_id", "dist"]].to_string())


# =============== Часть 2: иерархия (регионы) и уточнение ===============
print("\n\n######## PART 2: region-type search locations")
item_locs_all = set(items.item_location_id)
tr["s_is_region"] = ~tr.search_location_id.isin(item_locs_all)
print("rows with region-type search loc:", tr.s_is_region.mean())
print("match share for city-type search loc:", tr.loc[~tr.s_is_region, "match"].mean())
city_mm = tr[~tr.s_is_region & ~tr.match]
print("city-type mismatches: n=%d (%.4f of all rows); dist:" % (len(city_mm), len(city_mm) / len(tr)), q(city_mm.dist))
# регионы: разброс выбранных айтемов
reg = tr[tr.s_is_region].copy()
rc = reg.groupby("search_location_id").agg(rows=("lat", "size"), rlat=("lat", "median"), rlon=("lon", "median"),
                                           n_item_locs=("item_location_id", "nunique"))
reg = reg.join(rc[["rlat", "rlon"]], on="search_location_id")
reg["rd"] = hav(reg.lat, reg.lon, reg.rlat, reg.rlon)
rc = rc.join(reg.groupby("search_location_id").rd.agg(r50="median", r90=lambda x: x.quantile(.9)))
rc["top_item_loc_share"] = reg.groupby("search_location_id").item_location_id.agg(
    lambda s: s.value_counts(normalize=True).iloc[0])
print(rc.sort_values("rows", ascending=False).head(15).round(2).to_string())
print("region r50 (km) over regions with >=50 rows:", q(rc.loc[rc.rows >= 50, "r50"]))
# регионы в бенчмарке
bq["s_is_region"] = ~bq.search_location_id.isin(item_locs_all)
print("bench region-type share:", bq.s_is_region.mean(),
      "top:", bq.loc[bq.s_is_region, "search_location_id"].value_counts().head(8).to_dict())

# leak-free оценка: мапа search_loc -> item_locs по половине групп (query,loc), проверка на другой половине
rng = np.random.default_rng(0)
gkey = tr.search_query + "|" + tr.search_location_id.astype(str)
ug = gkey.unique()
test_g = set(rng.choice(ug, size=len(ug) // 2, replace=False))
is_test = gkey.isin(test_g)
fit, ev = tr[~is_test], tr[is_test]
m_cnt = fit.groupby(["search_location_id", "item_location_id"]).size().rename("c").reset_index()
m_cnt["share"] = m_cnt.c / m_cnt.groupby("search_location_id").c.transform("sum")
for thr in (0.0, 0.01, 0.03):
    allowed = set(map(tuple, m_cnt.loc[m_cnt.share > thr, ["search_location_id", "item_location_id"]].values))
    ok = [(s, i) in allowed or s == i for s, i in zip(ev.search_location_id.values, ev.item_location_id.values)]
    ok = np.array(ok)
    print(f"map filter (exact OR loc-pair share>{thr}): loss={1-ok.mean():.4f}; "
          f"loss region-type rows={1-ok[ev.s_is_region.values].mean():.4f}; city-type={1-ok[~ev.s_is_region.values].mean():.4f}")
# радиус от 'центроида' search loc, где для регионов центроид = медиана выбранных айтемов по fit-части
rcent = fit.groupby("search_location_id").agg(flat=("lat", "median"), flon=("lon", "median"))
ev = ev.join(rcent, on="search_location_id")
ev["slat"] = np.where(ev.s_clat.notna(), ev.s_clat, ev.flat)
ev["slon"] = np.where(ev.s_clon.notna(), ev.s_clon, ev.flon)
ev["d2"] = hav(ev.lat, ev.lon, ev.slat, ev.slon)
print("eval rows w/o any centroid:", ev.slat.isna().mean())
for R in (30, 100, 300):
    kept = ev.match | (ev.d2 <= R)
    print(f"R={R}: loss all={1-kept.mean():.4f} region-type={1-kept[ev.s_is_region].mean():.4f} "
          f"city-type={1-kept[~ev.s_is_region].mean():.4f}")

# ---- уточнённая эвристика городов: строгие окончания + фильтр по гео-согласованности ----
print("\n## Q4 refined")
ENDS = ("", "а", "е", "у", "ом", "ым", "ой", "и", "ы", "ах", "ам", "ского", "ском", "ской", "ске")
name_stem = {}
for name in city2loc:
    if name in STOP or " " in name or "-" in name or len(name) < 4:
        continue
    st = name[:-1] if name[-1] in "аеиоуыяьй" and len(name) >= 5 else name
    name_stem[name] = st
forms = {}
for name, st in name_stem.items():
    for e in ENDS:
        forms.setdefault(st + e, name)
    forms.setdefault(name, name)


def find_city2(query):
    qn = norm(query)
    for m in multi + [a for a in BIG if " " in a or "-" in a]:
        if m in qn:
            return m
    for t in TOK.findall(qn):
        if t in alias:
            return alias[t]
        if t in forms:
            return forms[t]
    return None


uq["city2"] = uq.search_query.map(find_city2)
tr = tr.merge(uq[["search_query", "city2"]], on="search_query", how="left")
# гео-согласованность имени: доля строк, где выбранный айтем в пределах 100 км от центроида города
name_cent = {n: (cent.clat.get(l), cent.clon.get(l)) for n, l in city2loc.items() if l is not None and l in cent.index}
t2 = tr[tr.city2.notna()].copy()
t2["nlat"] = t2.city2.map(lambda n: name_cent.get(n, (np.nan, np.nan))[0])
t2["nlon"] = t2.city2.map(lambda n: name_cent.get(n, (np.nan, np.nan))[1])
t2["dn"] = hav(t2.lat, t2.lon, t2.nlat, t2.nlon)
cons = t2.groupby("city2").agg(rows=("dn", "size"), near=("dn", lambda x: (x < 100).mean()))
bad = set(cons[(cons.rows >= 5) & (cons.near < 0.2)].index)
print("names dropped as non-geo (rows>=5, near<20%):", sorted(bad, key=lambda n: -cons.rows[n])[:40])
tr["city3"] = tr.city2.where(~tr.city2.isin(bad))
bq["city3"] = bq.search_query.map(find_city2)
bq["city3"] = bq.city3.where(~bq.city3.isin(bad))
print("REFINED share: bench", bq.city3.notna().mean(), " train rows", tr.city3.notna().mean(),
      " train uniq queries", tr.drop_duplicates("search_query").city3.notna().mean())
print("bench top:", bq.city3.value_counts().head(25).to_dict())
print("bench examples:", bq.loc[bq.city3.notna(), "search_query"].tolist()[:60])
wc = tr[tr.city3.notna()].copy()
print("train rows w/ city: match share=%.3f (vs no-city %.3f)" % (wc.match.mean(), tr[tr.city3.isna()].match.mean()))
wc["cloc"] = wc.city3.map(city2loc)
wc["nlat"] = wc.city3.map(lambda n: name_cent.get(n, (np.nan, np.nan))[0])
wc["nlon"] = wc.city3.map(lambda n: name_cent.get(n, (np.nan, np.nan))[1])
wc["dn"] = hav(wc.lat, wc.lon, wc.nlat, wc.nlon)
wc["d_named_vs_search"] = hav(wc.nlat, wc.nlon, wc.s_clat, wc.s_clon)
far = wc[wc.d_named_vs_search > 50]
print("city rows where named city is >50km from search loc (city-type search loc): n=%d (%.3f of city rows)"
      % (len(far), len(far) / len(wc)))
print("  of those: item in search loc %.3f; item within 50km of named city %.3f; item_loc==named loc %.3f"
      % (far.match.mean(), (far.dn < 50).mean(), (far.item_location_id == far.cloc).mean()))
reg_c = wc[wc.s_is_region]
print("city rows with region-type search loc: n=%d; item_loc==named loc %.3f; item within 50km of named %.3f"
      % (len(reg_c), (reg_c.item_location_id == reg_c.cloc).mean(), (reg_c.dn < 50).mean()))
print(far.sample(min(15, len(far)), random_state=0)[["search_query", "search_location_id", "item_location_id", "dist", "dn"]].round(0).to_string())
# маршрутные запросы (два города)
def two_cities(qs):
    qn = norm(qs); found = []
    for t in TOK.findall(qn):
        n = alias.get(t) or forms.get(t)
        if n and n not in bad and n not in found:
            found.append(n)
    return len(found) >= 2
print("bench queries with >=2 city names:", bq.search_query.map(two_cities).mean())
