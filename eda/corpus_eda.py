"""EDA slice: corpus, train<->corpus overlap, popularity, popularity baselines.
Loads only needed columns (memory-constrained machine)."""
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

D = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data") + "/"   # data/ в корне репозитория
pd.set_option("display.width", 200)


def q(s, ps=(0, .1, .25, .5, .75, .9, .95, .99, 1)):
    return s.quantile(list(ps)).round(2).to_dict()


def norm(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace("ё", "е").str.replace(r"\s+", " ", regex=True).str.strip()


print("=" * 30, "1. CORPUS")
it = pd.read_parquet(D + "benchmark_items.parquet",
                     columns=["item_id", "item_title_raw", "item_category_id", "item_microcat_id",
                              "item_price", "item_location_id"])
print("rows", len(it), "unique item_id", it.item_id.nunique(), "dups", len(it) - it.item_id.nunique())
print("category value counts:\n", it.item_category_id.value_counts().to_string())
mc = it.item_microcat_id.value_counts()
print("n microcats", len(mc), "items/microcat quantiles", q(mc))
print("top-20 microcats (count, share):")
print(pd.DataFrame({"n": mc.head(20), "share": (mc.head(20) / len(it)).round(4)}).to_string())
print("top-20 cumulative share", round(mc.head(20).sum() / len(it), 4))
# microcat -> category mapping purity
print("microcats spanning >1 category:", (it.groupby("item_microcat_id").item_category_id.nunique() > 1).sum())
loc = it.item_location_id.value_counts()
print("n locations", len(loc), "items/location quantiles", q(loc))
print("top-10 locations:\n", pd.DataFrame({"n": loc.head(10), "share": (loc.head(10) / len(it)).round(4)}).to_string())
print("top-10 loc cum share", round(loc.head(10).sum() / len(it), 4))
tw = it.item_title_raw.fillna("").str.split().str.len()
print("title words quantiles", q(tw), "mean", round(tw.mean(), 2))
tc = it.item_title_raw.fillna("").str.len()
print("title chars quantiles", q(tc))

# description length: single column via pyarrow, then drop
tbl = pq.read_table(D + "benchmark_items.parquet", columns=["item_description_raw"])
dl = pc.utf8_length(pc.fill_null(tbl.column("item_description_raw"), "")).to_pandas()
del tbl
print("desc null/empty share", round((dl == 0).mean(), 4), "<20 chars", round((dl < 20).mean(), 4),
      "<50", round((dl < 50).mean(), 4))
print("desc len quantiles", q(dl))
del dl

price = pd.to_numeric(it.item_price.astype(str), errors="coerce").astype(float)
it["price"] = price
print("price null share", round(price.isna().mean(), 4))
for v in [-1, 0, 1, 10, 100]:
    print(f"price=={v} share", round((price == v).mean(), 4))
print("price<=0 share", round((price <= 0).mean(), 4), "price<=1 share", round((price <= 1).mean(), 4), "price<100 share", round((price < 100).mean(), 4))
print("price quantiles", q(price.dropna()))
print("price quantiles excl <=1", q(price[price > 1]))
print("median price by category:\n", it.groupby("item_category_id").price.median().to_string())
print("top price values:\n", price.value_counts().head(10).to_string())

print("=" * 30, "2. TRAIN vs CORPUS")
tr = pd.read_parquet(D + "train.parquet",
                     columns=["search_query", "search_location_id", "item_id", "item_microcat_id",
                              "item_category_id", "item_location_id", "item_title_raw", "search_category"])
corpus_ids = set(it.item_id)
tr["in_corpus"] = tr.item_id.isin(corpus_ids)
tr_u = pd.Series(tr.item_id.unique())
tr_u_in = tr_u.isin(corpus_ids)
print("train rows", len(tr), "unique train items", len(tr_u))
print("share unique train items in corpus", round(tr_u_in.mean(), 4), "count", int(tr_u_in.sum()))
print("share train rows with item in corpus", round(tr.in_corpus.mean(), 4))
print("train item_location == search_location share", round((tr.item_location_id == tr.search_location_id).mean(), 4))
print("train item_category == search_category share", round((tr.item_category_id == tr.search_category).mean(), 4))
print("train item category counts:\n", tr.item_category_id.value_counts().head(10).to_string())
print("share of corpus items which appear in train", round(it.item_id.isin(set(tr_u)).mean(), 4))

# near-duplicates in corpus
it["tnorm"] = norm(it.item_title_raw)
g = it.groupby(["tnorm", "item_location_id"]).size()
print("corpus title+loc groups with >1 ids:", int((g > 1).sum()), "items in them:", int(g[g > 1].sum()),
      "share of corpus", round(g[g > 1].sum() / len(it), 4), "excess ids", int((g[g > 1] - 1).sum()))
g2 = it.groupby(["tnorm", "item_location_id", "price"]).size()
print("title+loc+price dup groups:", int((g2 > 1).sum()), "items:", int(g2[g2 > 1].sum()))
g3 = it.groupby("tnorm").size()
print("same title (any loc) dup groups:", int((g3 > 1).sum()), "items:", int(g3[g3 > 1].sum()),
      "share", round(g3[g3 > 1].sum() / len(it), 4))
print("largest title+loc groups:\n", g.sort_values(ascending=False).head(8).to_string())

print("=" * 30, "3. POPULARITY")
cnt = tr.item_id.value_counts()
print("items chosen once share (of unique)", round((cnt == 1).mean(), 4), "cnt quantiles", q(cnt))
for n in [50, 500, 5000, 50000]:
    print(f"top-{n} items cover rows share", round(cnt.head(n).sum() / len(tr), 4))
cc = cnt[cnt.index.isin(corpus_ids)]
print("corpus items ever chosen", len(cc), "share of corpus", round(len(cc) / len(it), 4),
      "never chosen share", round(1 - len(cc) / len(it), 4))
print("corpus-restricted: top-N cover of in-corpus rows:")
for n in [50, 500, 5000]:
    print(f"  top-{n}", round(cc.head(n).sum() / cc.sum(), 4))
print("top-10 popular items (title, count):")
tt = tr.drop_duplicates("item_id").set_index("item_id").item_title_raw
for i, c in cnt.head(10).items():
    print("  ", c, i in corpus_ids, tt[i][:60])

print("=" * 30, "query group stats")
tr["qn"] = norm(tr.search_query)
tr["gkey"] = tr.qn + "||" + tr.search_location_id.astype(str)
gs = tr.groupby("gkey").size()
print("n groups (query+loc)", len(gs), "rows/group quantiles", q(gs))
gu = tr.groupby("gkey").item_id.nunique()
print("unique items/group quantiles", q(gu))
print("n distinct normalized queries", tr.qn.nunique())
bq = pd.read_parquet(D + "benchmark_queries.parquet", columns=["search_query", "search_location_id"])
bq["qn"] = norm(bq.search_query)
bq["gkey"] = bq.qn + "||" + bq.search_location_id.astype(str)
print("benchmark queries whose normalized text is in train", round(bq.qn.isin(set(tr.qn)).mean(), 4),
      "whose query+loc in train", round(bq.gkey.isin(set(tr.gkey)).mean(), 4),
      "whose location in train", round(bq.search_location_id.isin(set(tr.search_location_id)).mean(), 4),
      "whose location in corpus", round(bq.search_location_id.isin(set(it.item_location_id)).mean(), 4))

print("=" * 30, "4. POPULARITY BASELINE LADDER")
rng = np.random.default_rng(42)
keys = gs.index.to_numpy()
hold = set(rng.choice(keys, size=int(round(0.1 * len(keys))), replace=False))
tr["hold"] = tr.gkey.isin(hold)
fit = tr[~tr.hold & tr.in_corpus]  # candidates restricted to corpus ids
te = tr[tr.hold]
print("held-out groups", len(hold), "rows", len(te))

glob = fit.item_id.value_counts().head(50).index.tolist()
loc_top = (fit.groupby(["search_location_id", "item_id"]).size().rename("n").reset_index()
           .sort_values(["search_location_id", "n"], ascending=[True, False])
           .groupby("search_location_id").head(50).groupby("search_location_id").item_id.agg(list).to_dict())
q_top = (fit.groupby(["qn", "item_id"]).size().rename("n").reset_index()
         .sort_values(["qn", "n"], ascending=[True, False])
         .groupby("qn").head(50).groupby("qn").item_id.agg(list).to_dict())


def fill(base, *others, k=50):
    out, seen = [], set()
    for lst in (base,) + others:
        for x in lst:
            if x not in seen:
                seen.add(x); out.append(x)
                if len(out) == k:
                    return out
    return out


te_g = te.groupby("gkey").agg(qn=("qn", "first"), loc=("search_location_id", "first"),
                              rel=("item_id", lambda s: set(s)),
                              rel_c=("item_id", lambda s: set(x for x in s if x in corpus_ids)))
res = {"a_global": [], "b_location": [], "c_query_hist": []}
res_c = {k: [] for k in res}
hist_hit = 0
for r in te_g.itertuples():
    ca = glob
    cb = fill(loc_top.get(r.loc, []), glob)
    qh = q_top.get(r.qn, [])
    hist_hit += bool(qh)
    ccands = fill(qh, cb)
    for name, cand in [("a_global", ca), ("b_location", cb), ("c_query_hist", ccands)]:
        s = set(cand)
        res[name].append(len(s & r.rel) / len(r.rel))
        if r.rel_c:
            res_c[name].append(len(s & r.rel_c) / len(r.rel_c))
print("held-out groups with query history in fit:", round(hist_hit / len(te_g), 4))
print("groups with >=1 corpus item:", len(res_c["a_global"]), "of", len(te_g))
for k in res:
    print(f"{k}: Recall@50 (rel=all chosen) {np.mean(res[k]):.4f} | (rel=chosen in corpus, groups w/ corpus item) {np.mean(res_c[k]):.4f}")
# c restricted to groups with history
hmask = np.array([bool(q_top.get(x)) for x in te_g.qn])
cvals = np.array(res["c_query_hist"])
bvals = np.array(res["b_location"])
print(f"c on groups with history: {cvals[hmask].mean():.4f}, without: {cvals[~hmask].mean():.4f}; b on same: {bvals[hmask].mean():.4f}/{bvals[~hmask].mean():.4f}")

print("=" * 30, "5. MICROCAT CONCENTRATION")
qm = tr.groupby(["qn", "item_microcat_id"]).size().rename("n").reset_index()
qtot = qm.groupby("qn").n.sum()
qmax = qm.groupby("qn").n.max()
qnm = qm.groupby("qn").size()
m2 = qtot >= 2
print("row-weighted share in top microcat (all texts)", round(qmax.sum() / qtot.sum(), 4))
print("row-weighted share in top microcat (texts with >=2 rows)", round(qmax[m2].sum() / qtot[m2].sum(), 4),
      "n texts", int(m2.sum()))
m5 = qtot >= 5
print("texts with >=5 rows: weighted", round(qmax[m5].sum() / qtot[m5].sum(), 4),
      "share of such texts with 1 microcat", round((qnm[m5] == 1).mean(), 4))
print("n microcats per text (>=2 rows) quantiles", q(qnm[m2]))
qc = tr.groupby(["qn", "item_category_id"]).size().rename("n").reset_index()
print("row-weighted share in top category (>=2 rows)",
      round(qc.groupby("qn").n.max()[m2].sum() / qc.groupby("qn").n.sum()[m2].sum(), 4))
# upper bound: if we knew the top microcat, how many candidates are in it in the corpus
print("corpus items per microcat median", mc.median(), "mean", round(mc.mean(), 1))
print("=" * 30, "6. LOCATION vs CORPUS SIZE (benchmark)")
it6 = pd.read_parquet(D + "benchmark_items.parquet", columns=["item_id", "item_location_id"])
tr6 = pd.read_parquet(D + "train.parquet", columns=["search_location_id", "item_location_id", "item_id"])
bq6 = pd.read_parquet(D + "benchmark_queries.parquet", columns=["search_location_id", "search_is_delivery_search"])
locn = it6.item_location_id.value_counts()
bq6["n_loc"] = bq6.search_location_id.map(locn).fillna(0)
print("benchmark: corpus items in query location quantiles", q(bq6.n_loc))
print("share benchmark queries with 0 corpus items in location", round((bq6.n_loc == 0).mean(), 4),
      "<=50", round((bq6.n_loc <= 50).mean(), 4), "<=500", round((bq6.n_loc <= 500).mean(), 4))
print("delivery share benchmark", round(bq6.search_is_delivery_search.mean(), 4))
inc = tr6.item_id.isin(set(it6.item_id))
same = tr6.item_location_id == tr6.search_location_id
print("train rows in corpus: item_loc==search_loc share", round(same[inc].mean(), 4),
      "; not in corpus:", round(same[~inc].mean(), 4))
