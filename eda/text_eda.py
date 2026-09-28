"""EDA slice: text matching query<->item and search filters.

Run: python eda/text_eda.py  (from the repository root). Prints all numbers used in eda/04_text_filters.md.
Memory: reads only needed columns; description read in batches and only for the sampled rows.
"""
import re
import gc
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = "data"
SEED = 42
N_SAMPLE = 50_000
rng = np.random.default_rng(SEED)

TOK_RE = re.compile(r"\w+")
STOP = set("в на и с для по от до из за к у о об не а или под при без".split())


def toks(s):
    if not isinstance(s, str) or not s:
        return []
    return TOK_RE.findall(s.lower().replace("ё", "е"))


def qtoks(s):
    # query tokens without stopwords / 1-char tokens
    return [t for t in toks(s) if t not in STOP and len(t) > 1]


def p5(ts):
    return {t[:5] for t in ts}


# ---------------------------------------------------------------- load
# two samples: (R) 50k random rows (popularity-weighted), (Q) one random row per unique query, 50k queries
# (Q) mimics macro-averaging over queries, which is what the benchmark does.
sq = pd.read_parquet(f"{DATA}/train.parquet", columns=["search_query"]).search_query
N = len(sq)
idx_r = np.sort(rng.choice(N, N_SAMPLE, replace=False))
perm = rng.permutation(N)
first = pd.Series(perm).groupby(sq.values[perm]).first().values  # random row per query
idx_q = np.sort(rng.choice(first, min(N_SAMPLE, len(first)), replace=False))
del sq, perm, first
all_idx = np.union1d(idx_r, idx_q)

# description only for sampled rows (read BEFORE other heavy columns to keep peak memory low)
pf = pq.ParquetFile(f"{DATA}/train.parquet")
desc = {}
off = 0
for b in pf.iter_batches(batch_size=50_000, columns=["item_description_raw"]):
    n = b.num_rows
    lo, hi = np.searchsorted(all_idx, [off, off + n])
    if hi > lo:
        for j, v in zip(all_idx[lo:hi], b.column(0).take(all_idx[lo:hi] - off).to_pylist()):
            desc[j] = v
    off += n
    del b
del pf
gc.collect()

tr_cols = ["search_query", "search_infm_params_text", "item_id", "item_title_raw",
           "item_infm_params_text", "item_rating"]
tr = pd.read_parquet(f"{DATA}/train.parquet", columns=tr_cols)


def make_sample(ix):
    s = tr.iloc[ix].reset_index(drop=True).copy()
    s["desc"] = [desc[j] for j in ix]
    return s


# ================================================================ Q1 lexical overlap
def overlap(sm, name):
    print("=" * 30, "Q1 lexical overlap:", name, len(sm))
    res = defaultdict(list)
    zero_rows = []
    for i, r in enumerate(sm.itertuples(index=False)):
        q = qtoks(r.search_query)
        if not q:
            res["empty_q"].append(1)
            continue
        res["empty_q"].append(0)
        T, P, D = set(toks(r.item_title_raw)), set(toks(r.item_infm_params_text)), set(toks(r.desc))
        qs = set(q)
        ex = {"t": bool(qs & T), "p": bool(qs & P), "d": bool(qs & D)}
        qp = p5(q)
        T5, P5, D5 = p5(T), p5(P), p5(D)
        st = {"t": bool(qp & T5), "p": bool(qp & P5), "d": bool(qp & D5)}
        for k in "tpd":
            res["ex_" + k].append(ex[k])
            res["st_" + k].append(st[k])
        res["ex_tp"].append(ex["t"] or ex["p"])
        res["st_tp"].append(st["t"] or st["p"])
        res["ex_any"].append(any(ex.values()))
        res["st_any"].append(any(st.values()))
        res["ex_all_tp"].append(qs <= (T | P))
        res["st_all_tp"].append(qp <= (T5 | P5))
        res["st_all_any"].append(qp <= (T5 | P5 | D5))
        res["st_all_t"].append(qp <= T5)
        res["st_cov_tp"].append(len(qp & (T5 | P5)) / len(qp))
        res["st_cov_tpd"].append(len(qp & (T5 | P5 | D5)) / len(qp))
        if not any(st.values()):
            zero_rows.append(i)

    print("queries with no content tokens:", np.mean(res["empty_q"]))
    tab = pd.DataFrame({
        "exact": [np.mean(res["ex_" + k]) for k in ["t", "p", "d", "tp", "any"]],
        "prefix5": [np.mean(res["st_" + k]) for k in ["t", "p", "d", "tp", "any"]],
    }, index=["title", "params", "description", "title+params", "any"])
    print(tab.round(4))
    print("ALL q tokens in title+params: exact", np.mean(res["ex_all_tp"]), "prefix5", np.mean(res["st_all_tp"]))
    print("ALL q tokens in title only (p5):", np.mean(res["st_all_t"]))
    print("ALL q tokens in title+params+desc (p5):", np.mean(res["st_all_any"]))
    print("mean coverage of q tokens (p5): tp", np.mean(res["st_cov_tp"]), "tpd", np.mean(res["st_cov_tpd"]))
    st_tp = np.array(res["st_tp"]); st_any = np.array(res["st_any"])
    print("desc-only rescue (p5, no match in t+p, match in d):", np.mean(~st_tp & st_any))
    print("zero-overlap share (p5, all fields):", 1 - st_any.mean(), "n=", len(zero_rows))
    z = sm.iloc[zero_rows]
    zs = z.sample(min(40, len(z)), random_state=SEED)
    print("--- zero overlap examples (query | title | filter)")
    for r in zs.itertuples(index=False):
        print(f"{r.search_query!r} -> {r.item_title_raw!r} | {r.search_infm_params_text[:60]!r}")


sm = make_sample(idx_r)
overlap(sm, "R: random rows")
smq = make_sample(idx_q)
overlap(smq, "Q: one row per unique query")
del smq, desc
gc.collect()

# ================================================================ Q2 query stats
print("=" * 30, "Q2 query stats")
bq = pd.read_parquet(f"{DATA}/benchmark_queries.parquet")
bi_cols = ["item_title_raw", "item_infm_params_text"]
bi = pd.read_parquet(f"{DATA}/benchmark_items.parquet", columns=bi_cols)

title_vocab = set()
for s in bi.item_title_raw.values:
    title_vocab.update(toks(s))
corpus_title_vocab = set(title_vocab)
for s in tr.item_title_raw.drop_duplicates().values:
    title_vocab.update(toks(s))
tp_vocab = set(title_vocab)
for s in bi.item_infm_params_text.values:
    tp_vocab.update(toks(s))
tp5 = {t[:5] for t in tp_vocab}
print("vocab sizes: corpus titles", len(corpus_title_vocab), "titles(corpus+train)", len(title_vocab),
      "+corpus params", len(tp_vocab))

# query vocab from train queries (to see if OOV-in-titles tokens are at least seen in other queries)
trq_counts = Counter()
for s in tr.search_query.values:
    trq_counts.update(set(toks(s)))


def qstats(qs, name, dedup):
    qs = pd.Series(qs)
    if dedup:
        qs = qs.drop_duplicates()
    ntok = qs.map(lambda s: len(toks(s)))
    lat = qs.str.contains(r"[A-Za-z]", regex=True)
    dig = qs.str.contains(r"\d", regex=True)

    def oov(s, vocab):
        return any((t not in vocab) and not t.isdigit() for t in toks(s))

    def oov5(s):
        return any((t[:5] not in tp5) and not t.isdigit() for t in toks(s))

    def oov_rare(s):  # not in title vocab AND rare among train queries (<=2 queries)
        return any((t not in title_vocab) and not t.isdigit() and trq_counts.get(t, 0) <= 2 for t in toks(s))

    out = {
        "n": len(qs),
        "tokens_mean": ntok.mean(), "tokens_median": ntok.median(),
        "1tok": (ntok == 1).mean(), "2tok": (ntok == 2).mean(), "3tok": (ntok == 3).mean(),
        "4+tok": (ntok >= 4).mean(),
        "latin": lat.mean(), "digits": dig.mean(),
        "oov_corpus_titles": qs.map(lambda s: oov(s, corpus_title_vocab)).mean(),
        "oov_all_titles": qs.map(lambda s: oov(s, title_vocab)).mean(),
        "oov_titles+params": qs.map(lambda s: oov(s, tp_vocab)).mean(),
        "oov_prefix5_t+p": qs.map(oov5).mean(),
        "oov_rare(typo-like)": qs.map(oov_rare).mean(),
    }
    return pd.Series(out, name=name)


print("train unique queries:", tr.search_query.nunique(), "rows/query:", N / tr.search_query.nunique())
qst = pd.concat([
    qstats(sm.search_query.values, "train_rows(sample)", False),
    qstats(tr.search_query.values, "train_unique", True),
    qstats(bq.search_query.values, "benchmark", False),
], axis=1)
print(qst.round(4).to_string())
# examples of typo-like tokens
ex_oov = Counter()
for s in bq.search_query.values:
    for t in toks(s):
        if t not in title_vocab and not t.isdigit() and trq_counts.get(t, 0) <= 2:
            ex_oov[t] += 1
print("benchmark typo-like tokens examples:", list(ex_oov)[:60])
bq_in_train = bq.search_query.isin(set(tr.search_query.values)).mean()
bq_in_train_norm = bq.search_query.map(lambda s: " ".join(toks(s))).isin(
    set(" ".join(toks(s)) for s in tr.search_query.drop_duplicates().values)).mean()
print("benchmark query text seen in train: exact", bq_in_train, "normalized", bq_in_train_norm)

# ================================================================ Q3 filters
print("=" * 30, "Q3 filters")
KEYS = ["Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Онлайн-запись", "Онлайн-бронирование",
        "Кто оказывает услуги", "Срочная услуга (мультистатус)", "Рейтинг пользователя", "Поиск по слотам",
        "Сортировка для URL", "Где вы оказываете услуги", "Предмет или специальность", "Чем вы занимаетесь",
        "Ваши клиенты", "Специальность или сфера", "Слова в описании", "Услуги", "Услуга",
        "Участие в пилоте CPL", "Аренда авто", "Гарантия", "Коробка передач", "Класс авто"]
KEY_RE = re.compile("(" + "|".join(re.escape(k) for k in sorted(KEYS, key=len, reverse=True)) + r")(?=\s|$)")


def parse_filters(s):
    """-> list of (key, value)."""
    if not s:
        return []
    parts = []
    ms = list(KEY_RE.finditer(s))
    if not ms:
        return [("<other>", s.strip())]
    if ms[0].start() > 0:
        parts.append(("<other>", s[: ms[0].start()].strip()))
    # a key-like word right after "Вид/Тип услуги" is actually the value ("Тип услуги Аренда авто",
    # "Вид услуги Услуги эвакуатора") -> drop such matches
    keep = []
    for m in ms:
        if keep and keep[-1].group(1) in ("Вид услуги", "Тип услуги", "Тип услуги автосервиса") \
                and not s[keep[-1].end():m.start()].strip() and m.group(1) in ("Услуги", "Услуга", "Аренда авто"):
            continue
        keep.append(m)
    ms = keep
    for j, m in enumerate(ms):
        end = ms[j + 1].start() if j + 1 < len(ms) else len(s)
        parts.append((m.group(1), s[m.end():end].strip()))
    return parts


fs = tr.search_infm_params_text.fillna("")
print("train non-empty filter share:", (fs != "").mean(), "distinct strings:", fs.nunique())
print("top-20 full filter strings (train rows):")
vc = fs.value_counts()
print((vc.head(21) / N).round(4).to_string())

uniq = vc.index.tolist()
parsed = {s: parse_filters(s) for s in uniq}


def key_mix(series):
    c = Counter()
    n = len(series)
    for s, cnt in series.value_counts().items():
        for k in {k for k, v in parsed.get(s, parse_filters(s))}:
            c[k] += cnt
    return pd.Series(c).sort_values(ascending=False) / n


bfs = bq.search_infm_params_text.fillna("")
for s in bfs.unique():
    if s not in parsed:
        parsed[s] = parse_filters(s)
mix = pd.concat([key_mix(fs).rename("train"), key_mix(bfs).rename("benchmark")], axis=1).fillna(0)
print("filter key presence (share of all rows):")
print(mix.round(4).to_string())
print("rating filter values:", Counter(v for s in uniq for k, v in parsed[s] if k == "Рейтинг пользователя"))
print("'<other>' examples:", [v for s in uniq for k, v in parsed[s] if k == "<other>"][:15])

# ---- compatibility with chosen item (full train, filters non-empty)
print("--- compatibility (full train)")
RATING_RE = re.compile(r"(\d)")
mask = fs != ""
sub = tr.loc[mask, ["search_infm_params_text", "item_infm_params_text", "item_rating"]]
viol = defaultdict(lambda: [0, 0, 0])  # key -> [n_rows_with_key, n_violating, n_nan/undetermined]
any_hard = [0, 0]
per_value_viol = defaultdict(lambda: [0, 0])
for s, ip, rating in sub.itertuples(index=False):
    ip = ip or ""
    groups = defaultdict(list)
    for k, v in parsed[s]:
        groups[k].append(v)
    row_violates_core = False
    for k, vals in groups.items():
        if k == "Рейтинг пользователя":
            thr = min(int(RATING_RE.search(v).group(1)) for v in vals if RATING_RE.search(v))
            viol[k][0] += 1
            if rating is None or np.isnan(rating):
                viol[k][2] += 1
            elif rating < thr:
                viol[k][1] += 1
            continue
        vals = [v for v in vals if v]
        if k == "Участие в пилоте CPL" or (k == "Кто оказывает услуги" and vals and
                                              set(vals) <= {"Частный исполнитель", "Компания"}):
            viol[k + " (тип продавца/CPL: нет в params)"][0] += 1
            viol[k + " (тип продавца/CPL: нет в params)"][2] += 1
            continue
        if k in ("Онлайн-запись", "Онлайн-бронирование", "Гарантия"):
            viol[k][0] += 1
            ok = k in ip
            viol[k][1] += (not ok)
            continue
        if not vals:
            viol[k + " (пустое значение)"][0] += 1
            continue
        if k in ("Сортировка для URL", "Поиск по слотам", "Срочная услуга (мультистатус)", "Слова в описании", "<other>"):
            viol[k][0] += 1
            viol[k][2] += 1  # not checkable against item params
            continue
        viol[k][0] += 1
        ok = any(f"{k} {v}" in ip for v in vals)
        if not ok:
            viol[k][1] += 1
            if k in ("Вид услуги", "Тип услуги", "Тип услуги автосервиса"):
                row_violates_core = True
        for v in vals:
            per_value_viol[(k, v)][0] += 1
            per_value_viol[(k, v)][1] += (f"{k} {v}" not in ip)
    if any(k in groups for k in ("Вид услуги", "Тип услуги", "Тип услуги автосервиса")):
        any_hard[0] += 1
        any_hard[1] += row_violates_core

vt = pd.DataFrame(viol, index=["rows", "violating", "undetermined/NaN"]).T
vt["viol_share"] = vt.violating / vt.rows
vt["undet_share"] = vt["undetermined/NaN"] / vt.rows
vt["rows_share_of_train"] = vt.rows / N
print(vt.sort_values("rows", ascending=False).round(4).to_string())
print("rows with Вид/Тип filter:", any_hard[0], "cut by hard Вид+Тип filter:", any_hard[1] / any_hard[0])
pv = pd.DataFrame([(k, v, a, b, b / a) for (k, v), (a, b) in per_value_viol.items() if a >= 1500],
                  columns=["key", "value", "rows", "viol", "viol_share"]).sort_values("viol_share", ascending=False)
# why Вид услуги violated: item has other Вид vs no Вид
VID_RE = re.compile(r"Вид услуги (.*?)(?= Место оказания| Тип услуги| Тип стоимости|$)")
why = Counter()
for s_, ip in sub[["search_infm_params_text", "item_infm_params_text"]].itertuples(index=False):
    vals = [v for k, v in parsed[s_] if k == "Вид услуги" and v]
    if vals and not any(f"Вид услуги {v}" in (ip or "") for v in vals):
        m = VID_RE.search(ip or "")
        why["item_vid_empty_or_missing" if (not m or not m.group(1).strip()) else "item_other_vid"] += 1
print("Вид услуги violation reasons:", why)
print("per-value violation (>=1500 rows):")
print(pv.round(3).to_string())

# rating distribution nuance
r = tr.item_rating
print("item_rating: NaN", r.isna().mean(), "<4", (r < 4).mean(), "==0", (r == 0).mean())

# examples of violations for Вид услуги
ex = []
for s, ip, q in tr.loc[mask, ["search_infm_params_text", "item_infm_params_text", "search_query"]].sample(
        20000, random_state=SEED).itertuples(index=False):
    g = [(k, v) for k, v in parsed[s] if k == "Вид услуги" and v]
    if g and not any(f"Вид услуги {v}" in (ip or "") for k, v in g):
        m = re.search(r"Вид услуги (.+?) (Место|Тип|$)", ip or "")
        ex.append((q, g[0][1], m.group(1) if m else (ip or "")[:40]))
print("Вид услуги violations examples (query | filter | item vid):")
for e in ex[:15]:
    print(e)

# ================================================================ Q4 benchmark filters
print("=" * 30, "Q4 benchmark filters")
print("benchmark non-empty filter share:", (bfs != "").mean())
print("benchmark top-15 filter strings:")
print((bfs.value_counts().head(16) / len(bfs)).round(4).to_string())
print("share of benchmark filter strings seen in train:", bfs[bfs != ""].isin(set(uniq)).mean())

# selectivity: how much of the corpus a hard "Вид услуги" filter keeps (benchmark queries)
bip = pd.read_parquet(f"{DATA}/benchmark_items.parquet", columns=["item_infm_params_text"]).item_infm_params_text.fillna("")
cvid = bip.str.extract(r"Вид услуги (.*?)(?= Место оказания| Тип услуги| Тип стоимости|$)")[0].fillna("").value_counts(normalize=True)
print("corpus Вид услуги distribution (top 12):")
print(cvid.head(12).round(4).to_string())
kept = []
for s_ in bfs[bfs != ""]:
    vals = [v for k, v in parsed[s_] if k == "Вид услуги" and v]
    if vals:
        kept.append(sum(cvid.get(v, 0) for v in vals))
print("benchmark queries with non-empty Вид filter:", len(kept), "mean corpus share kept:", np.mean(kept),
      "median:", np.median(kept))
