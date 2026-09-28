"""Сводка сравнения stem / lemma / lemma+stem по результатам experiments/run_sparse_lemma.py.

python experiments/lemma_report.py
Печатает таблицу, парный бутстреп, кривую и срезы лучшей новой схемы, разбор расхождений
lemma vs stem; копирует кандидатов лучшей новой схемы в work/cands/sparse_lemma_*.parquet.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sparse import _STEMMER, _TOKEN_RE, WORK, doc_text, load_items  # noqa: E402
from sparse_lemma import morph  # noqa: E402
from validation import load_validation, slices  # noqa: E402

LEM = WORK / "lemma"
B = 10_000


def boot(diff: np.ndarray, seed: int = 0) -> tuple:
    """Парный бутстреп по запросам: среднее разности и 95% интервал."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    means = diff[idx].mean(1)
    return diff.mean(), *np.percentile(means, [2.5, 97.5])


def words(t: str) -> list[str]:
    return _TOKEN_RE.findall(str(t).lower().replace("ё", "е"))


def lem(w: str) -> str:
    return morph().parse(w)[0].normal_form.replace("ё", "е")


def main():
    perq = pd.read_parquet(LEM / "perq.parquet")
    timing = pd.read_csv(LEM / "timing.csv")
    val_q, labels, _ = load_validation()
    wide = perq.pivot_table(index=["query_id"], columns=["scheme", "mode", "k"], values="recall")

    print("== Recall@50")
    t = perq[perq.k == 50].groupby(["scheme", "mode"])["recall"].mean().unstack()
    print(t.round(4).to_string())
    print("\n== Время и словарь")
    print(timing.round(3).to_string())

    print("\n== Парный бутстреп, разность со stem (95% ДИ)")
    for sch in ("lemma", "lemma+stem"):
        for mode in ("all", "geo"):
            for k in (50, 300):
                d = (wide[(sch, mode, k)] - wide[("stem", mode, k)]).to_numpy()
                m, lo, hi = boot(d)
                print(f"{sch:>10} {mode} @{k}: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  "
                      f"лучше/хуже по запросам: {(d > 0).sum()}/{(d < 0).sum()}")

    best = max(("lemma", "lemma+stem"), key=lambda s: t.loc[s, "geo"])
    print(f"\n== Лучшая новая схема: {best}. Кривая")
    for sch in ("stem", best):
        for mode in ("all", "geo"):
            print(sch, mode, " ".join(f"@{k}={wide[(sch, mode, k)].mean():.4f}" for k in (50, 100, 200, 300)))

    print("\n== Срезы, Recall@50")
    cols = {}
    for sch in ("stem", best):
        for mode in ("all", "geo"):
            s = slices(wide[(sch, mode, 50)], val_q)
            cols[f"{sch} {mode}"] = s["recall"]
            n = s["запросов"]
    sl = pd.DataFrame(cols)
    sl.insert(0, "запросов", n.astype(int))
    sl[f"Δ geo"] = sl[f"{best} geo"] - sl["stem geo"]
    print(sl.round(4).to_string())

    # ---------------------------------------------------------------- расхождения lemma vs stem (geo)
    print("\n== Расхождения lemma vs stem, geo @50 (есть хоть один правильный в топ-50)")
    hs, hl = wide[("stem", "geo", 50)] > 0, wide[("lemma", "geo", 50)] > 0
    only_l, only_s = hl & ~hs, hs & ~hl
    print(f"находит только lemma: {only_l.sum()}, только stem: {only_s.sum()}")
    cs = pd.read_parquet(LEM / "stem_val_geo.parquet")
    cl = pd.read_parquet(LEM / "lemma_val_geo.parquet")
    items = load_items()
    items = items.set_index(items["item_id"].astype(str))
    qtext = val_q.set_index("query_id")["search_query"]
    rng = np.random.default_rng(1)
    pick = list(rng.choice(only_l[only_l].index, 5, replace=False)) + \
        list(rng.choice(only_s[only_s].index, 5, replace=False))
    for qid in pick:
        rel = labels[qid]
        q = qtext[qid]
        rs = cs[(cs.query_id == qid) & cs.item_id.isin(rel)]["rank"].min()
        rl = cl[(cl.query_id == qid) & cl.item_id.isin(rel)]["rank"].min()
        ans = [i for i in rel if i in items.index][0]
        doc = doc_text(items.loc[[ans]], "b").iloc[0]
        dw = set(words(doc))
        dstem = set(_STEMMER.stemWords(list(dw)))
        dlem = {lem(w) for w in dw}
        who = "lemma" if qid in only_l[only_l].index else "stem"
        print(f"\n[{who}] q={q!r}  ранг stem={rs}, lemma={rl}")
        print("   ответ:", items.loc[ans, "item_title_raw"])
        for w in dict.fromkeys(words(q)):
            s, l_ = _STEMMER.stemWord(w), lem(w)
            ds = sorted(x for x in dw if _STEMMER.stemWord(x) == s)[:4]
            dl = sorted(x for x in dw if lem(x) == l_)[:4]
            print(f"   {w}: stem={s} {'+' if s in dstem else '-'}{ds}  lemma={l_} {'+' if l_ in dlem else '-'}{dl}")
        top_s = cs[(cs.query_id == qid) & (cs["rank"] <= 3)].item_id
        top_l = cl[(cl.query_id == qid) & (cl["rank"] <= 3)].item_id
        print("   топ-3 stem :", " | ".join(items.loc[top_s, "item_title_raw"].str.slice(0, 50)))
        print("   топ-3 lemma:", " | ".join(items.loc[top_l, "item_title_raw"].str.slice(0, 50)))

    # ---------------------------------------------------------------- выгрузка кандидатов
    out = WORK / "cands"
    for part in ("val", "bench"):
        for mode in ("all", "geo"):
            dst = out / f"sparse_lemma_{part}_{mode}.parquet"
            shutil.copyfile(LEM / f"{best}_{part}_{mode}.parquet", dst)
            c = pd.read_parquet(dst)
            print(dst.name, len(c), "строк,", c["query_id"].nunique(), "запросов,", dict(c.dtypes.astype(str)))


if __name__ == "__main__":
    main()
