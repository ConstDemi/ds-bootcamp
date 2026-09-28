"""EDA: структура запросов в train и пересечение train <-> benchmark."""
import re
import numpy as np
import pandas as pd

D = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data") + "/"   # data/ в корне репозитория
QCOLS = ["search_query", "search_location_id", "search_is_delivery_search",
         "search_infm_params_text", "search_category"]

tr = pd.read_parquet(D + "train.parquet", columns=QCOLS + ["item_id"])
bq = pd.read_parquet(D + "benchmark_queries.parquet")
bi_ids = pd.read_parquet(D + "benchmark_items.parquet", columns=["item_id"])["item_id"]
bi_set = set(bi_ids.tolist())

ws = re.compile(r"\s+")


def norm(s):
    return s.fillna("").astype(str).str.lower().str.strip().str.replace(ws, " ", regex=True)


for df in (tr, bq):
    df["qn"] = norm(df["search_query"])
    df["inf"] = df["search_infm_params_text"].fillna("").astype(str)
    df["loc"] = df["search_location_id"].astype(str)
    df["dlv"] = df["search_is_delivery_search"].astype(str)
    df["k_raw"] = df["search_query"].astype(str)
    df["k_tl"] = df["qn"] + "|" + df["loc"]
    df["k_full"] = df["qn"] + "|" + df["loc"] + "|" + df["dlv"] + "|" + df["inf"]
    df["k_tli"] = df["qn"] + "|" + df["loc"] + "|" + df["inf"]

print("=== dtypes ===")
print(tr.dtypes)
print(bq.dtypes)
print("train rows", len(tr), "bench queries", len(bq), "bench items", len(bi_ids), "unique", bi_ids.nunique())

print("\n=== Q1: число групп по ключам ===")
for k in ["k_raw", "qn", "k_tl", "k_tli", "k_full"]:
    print(k, "train uniq:", tr[k].nunique(), " bench uniq:", bq[k].nunique())
print("bench rows", len(bq), "bench query_id uniq", bq["query_id"].nunique())
# raw text + loc + dlv + inf (without normalization)
raw_full = tr["k_raw"] + "|" + tr["loc"] + "|" + tr["dlv"] + "|" + tr["inf"]
print("raw_full train uniq:", raw_full.nunique())

# does normalization merge anything?
print("raw texts that collapse under norm:", tr["k_raw"].nunique() - tr["qn"].nunique())


def dist(sizes, name):
    s = sizes
    print(f"{name}: n_groups={len(s)} mean={s.mean():.2f} median={s.median()} p75={s.quantile(.75)} "
          f"p90={s.quantile(.9)} p99={s.quantile(.99)} max={s.max()} share1={(s==1).mean():.3f} "
          f"share>=50={(s>=50).mean():.4f}")


for k in ["qn", "k_tl", "k_full"]:
    dist(tr.groupby(k).size(), "rows/group " + k)
    dist(tr.groupby(k)["item_id"].nunique(), "uniq items/group " + k)

print("\n=== дубликаты (ключ + item_id) ===")
for k in ["qn", "k_tl", "k_full"]:
    d = tr.duplicated([k, "item_id"]).sum()
    print(k, "dup rows:", d, f"({d/len(tr):.4f})")
d_all = tr.duplicated(QCOLS + ["item_id"]).sum()
print("exact dup on all query cols + item_id:", d_all)

# consistency: does a key determine delivery/infm/category?
g = tr.groupby("k_tl")
print("k_tl groups with >1 delivery value:", (g["dlv"].nunique() > 1).sum())
print("k_tl groups with >1 infm value:", (g["inf"].nunique() > 1).sum())

print("\n=== Q2 ===")
print("train search_category:\n", tr["search_category"].value_counts(dropna=False).head(10))
print("bench search_category:\n", bq["search_category"].value_counts(dropna=False).head(10))
print("delivery share train rows:", tr["search_is_delivery_search"].astype(float).mean())
tu = tr.drop_duplicates("k_full")
print("delivery share train groups(k_full):", tu["search_is_delivery_search"].astype(float).mean())
print("delivery share bench:", bq["search_is_delivery_search"].astype(float).mean())
print("empty infm train rows:", (tr["inf"] == "").mean(), " groups:", (tu["inf"] == "").mean(),
      " bench:", (bq["inf"] == "").mean())
print("top infm values train groups:\n", tu.loc[tu.inf != "", "inf"].value_counts().head(10))
print("top infm values bench:\n", bq.loc[bq.inf != "", "inf"].value_counts().head(10))
print("uniq locations train:", tr["loc"].nunique(), " bench:", bq["loc"].nunique(),
      " bench locs in train:", bq["loc"].isin(set(tr["loc"])).mean())
print("top loc train groups:\n", tu["loc"].value_counts(normalize=True).head(5))
print("top loc bench:\n", bq["loc"].value_counts(normalize=True).head(5))

print("\n=== Q3: overlap ===")
for k in ["k_raw", "qn", "k_tl", "k_tli", "k_full"]:
    s = set(tr[k])
    print(k, "bench share in train:", round(bq[k].isin(s).mean(), 4))

# word lengths
def wlen(s):
    return s.str.split().str.len()

tr["wl"] = wlen(tr["qn"])
bq["wl"] = wlen(bq["qn"])
tut = tr.drop_duplicates("qn")
for name, s in [("train rows", tr["wl"]), ("train uniq texts", tut["wl"]),
                ("train groups k_full", tu.assign(wl=wlen(tu["qn"]))["wl"]), ("bench", bq["wl"])]:
    vc = s.clip(upper=6).value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"{name}: mean={s.mean():.2f} median={s.median()} dist(<=6+)={vc}")
print("char len train uniq:", tut["qn"].str.len().describe().round(1).to_dict())
print("char len bench:", bq["qn"].str.len().describe().round(1).to_dict())

print("\ntop20 train (rows):\n", tr["qn"].value_counts().head(20))
print("\ntop20 train (k_full groups):\n", tu["qn"].value_counts().head(20))
print("\ntop20 bench:\n", bq["qn"].value_counts().head(20))

# frequency of bench texts in train
txt_rows = tr["qn"].value_counts()
txt_groups = tu["qn"].value_counts()
bq["tr_rows"] = bq["qn"].map(txt_rows).fillna(0).astype(int)
bq["tr_groups"] = bq["qn"].map(txt_groups).fillna(0).astype(int)
bins = [-1, 0, 1, 4, 19, 99, 10**9]
labels = ["0", "1", "2-4", "5-19", "20-99", "100+"]
print("\nbench texts by #train rows with same text:\n",
      pd.cut(bq["tr_rows"], bins, labels=labels).value_counts(normalize=True).sort_index().round(3))
print("bench texts by #train groups(k_full) with same text:\n",
      pd.cut(bq["tr_groups"], bins, labels=labels).value_counts(normalize=True).sort_index().round(3))
# reference: same thing for a random sample of train groups (text frequency among OTHER groups)
rng = np.random.default_rng(0)
samp = tu.sample(len(bq), random_state=0)
other_groups = samp["qn"].map(txt_groups) - 1
print("REFERENCE random train groups: #other groups with same text:\n",
      pd.cut(other_groups, bins, labels=labels).value_counts(normalize=True).sort_index().round(3))
# reference weighted by rows? random sample of rows -> text; count other rows
samp_rows = tr.sample(len(bq), random_state=0)
print("REFERENCE random train rows: #train rows with same text (incl itself):\n",
      pd.cut(samp_rows["qn"].map(txt_rows), bins, labels=labels).value_counts(normalize=True).sort_index().round(3))

# frequency of text in train uniq-groups ranking: popularity percentile of bench texts
print("\nshare of train groups(k_full) whose text is 'singleton text' (1 group):",
      (tu["qn"].map(txt_groups) == 1).mean())
print("share of bench texts whose text appears in exactly 0 train groups:", (bq["tr_groups"] == 0).mean())

# Is bench duplicated inside itself?
print("bench duplicate qn:", bq["qn"].duplicated().sum(), " dup k_full:", bq["k_full"].duplicated().sum())

print("\n=== Q5: history as candidates ===")
seen = bq[bq["tr_rows"] > 0].copy()
items_per_text = tr.groupby("qn")["item_id"].agg(lambda x: set(x))
seen["hist_items"] = seen["qn"].map(items_per_text)
seen["n_hist"] = seen["hist_items"].str.len()
seen["n_hist_in_corpus"] = seen["hist_items"].map(lambda s: sum(i in bi_set for i in s))
seen["share_in_corpus"] = seen["n_hist_in_corpus"] / seen["n_hist"]
print("bench queries with seen text:", len(seen))
print("n distinct hist items per text:", seen["n_hist"].describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict())
print("n hist items in corpus:", seen["n_hist_in_corpus"].describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict())
print("share in corpus (mean over queries):", seen["share_in_corpus"].mean().round(3),
      " median:", seen["share_in_corpus"].median())
print("queries with >=1 hist item in corpus:", (seen["n_hist_in_corpus"] > 0).mean().round(3))
print("queries with >=50 hist items in corpus:", (seen["n_hist_in_corpus"] >= 50).mean().round(3))
# pooled
all_hist = set().union(*seen["hist_items"].tolist())
print("pooled distinct hist items:", len(all_hist), " in corpus:", sum(i in bi_set for i in all_hist))
# with k_full key
items_per_full = tr.groupby("k_full")["item_id"].agg(lambda x: set(x))
sf = bq[bq["k_full"].isin(items_per_full.index)].copy()
sf["hist"] = sf["k_full"].map(items_per_full)
sf["n_in"] = sf["hist"].map(lambda s: sum(i in bi_set for i in s))
print("k_full seen:", len(sf), " n hist items in corpus mean/median:", sf["n_in"].mean().round(2), sf["n_in"].median())

# global: share of train item_ids in corpus
tr_items = pd.Series(tr["item_id"].unique())
print("\ntrain uniq items:", len(tr_items), " in bench corpus:", tr_items.isin(bi_set).mean().round(4),
      " abs:", tr_items.isin(bi_set).sum())
print("bench corpus items that appear in train:", bi_ids.isin(set(tr_items)).mean().round(4))
print("train rows whose item in corpus:", tr["item_id"].isin(bi_set).mean().round(4))

print("\n=== Q4: симуляция схем сэмплирования бенчмарка ===")
gpt = tu.groupby("qn").size()  # k_full groups per text
print("groups per unique text: share1=", (gpt == 1).mean().round(3), " share>=2=", (gpt >= 2).mean().round(3))


def simulate(scheme, seed):
    r = np.random.default_rng(seed)
    if scheme == "uniform_text":  # uniform over distinct texts, hold out one k_full group of it
        texts = r.choice(gpt.index.values, len(bq), replace=False)
        cand = tu[tu["qn"].isin(set(texts))]
        held = cand.groupby("qn").sample(1, random_state=seed)
    elif scheme == "uniform_group":  # uniform over k_full groups
        held = tu.sample(len(bq), random_state=seed)
    elif scheme == "uniform_row":  # weighted by rows (clicks)
        held = tr.sample(len(bq), random_state=seed).drop_duplicates("k_full")
    rest = tr[~tr["k_full"].isin(set(held["k_full"]))]
    out = {
        "text_in_rest": held["qn"].isin(set(rest["qn"])).mean(),
        "text+loc_in_rest": held["k_tl"].isin(set(rest["k_tl"])).mean(),
        "mean_words": held["qn"].str.split().str.len().mean(),
        "empty_infm": (held["inf"] == "").mean(),
        "dup_text_in_sample": held["qn"].duplicated().mean(),
    }
    fr = held["qn"].map(rest["qn"].value_counts()).fillna(0)
    b = pd.cut(fr, bins, labels=labels).value_counts(normalize=True).sort_index().round(3)
    out.update({f"rows_{k}": v for k, v in b.items()})
    return out


rows = {}
for sch in ["uniform_text", "uniform_group", "uniform_row"]:
    rows[sch] = pd.DataFrame([simulate(sch, s) for s in range(3)]).mean()
act = {"text_in_rest": (bq["tr_rows"] > 0).mean(), "text+loc_in_rest": bq["k_tl"].isin(set(tr["k_tl"])).mean(),
       "mean_words": bq["wl"].mean(), "empty_infm": (bq["inf"] == "").mean(),
       "dup_text_in_sample": bq["qn"].duplicated().mean()}
b = pd.cut(bq["tr_rows"], bins, labels=labels).value_counts(normalize=True).sort_index().round(3)
act.update({f"rows_{k}": v for k, v in b.items()})
rows["BENCHMARK"] = pd.Series(act)
print(pd.DataFrame(rows).round(3).to_string())

print("\n=== доп: category 0 / infm в бенчмарке ===")
c0 = bq[bq["search_category"] == 0]
print("bench cat0 n=", len(c0), " empty infm share:", (c0["inf"] == "").mean().round(3),
      " text seen in train:", (c0["tr_rows"] > 0).mean().round(3))
print(c0["qn"].head(15).tolist())
c114 = bq[bq["search_category"] == 114]
print("bench cat114 empty infm share:", (c114["inf"] == "").mean().round(3))
print("train cat0 rows:", tr.loc[tr.search_category == 0, ["qn", "inf"]].head(10).to_string())
# infm for bench vs train by same text
st = bq[bq["tr_rows"] > 0]
print("bench seen-text: empty infm share", (st["inf"] == "").mean().round(3))
# location: first-occurrence per unique text in train vs bench
print("top loc train unique texts:\n", tut["loc"].value_counts(normalize=True).head(5).round(3))
