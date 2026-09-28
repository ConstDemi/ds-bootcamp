"""Cross-encoder реранкер над кандидатами (см. eda/12_rerank.md).

Вход — кандидаты любого ретривера в общем формате (query_id, item_id str, score, rank с 1),
таблица запросов (query_id, search_query, search_infm_params_text) и таблица объявлений
(item_id, item_title_raw, item_infm_params_text, item_description_raw).
Выход — те же кандидаты в том же формате: первые k переставлены, хвост (rank > k) остался как был.
score у переставленной части = скор слияния, у хвоста — исходный; порядок задаёт только rank.

Схема:
1. пара = (запрос + значения фильтров, документ item_text), документ обрезан по токенам так,
   чтобы пара влезала в max_len (256) токенов;
2. cross-encoder считает скор каждой пары из топ-k (GPU fp16, пары отсортированы по длине —
   меньше паддинга); скоры кешируются в parquet (query_id, item_id, ce), повторный запуск
   с другим k или слиянием считает только новые пары;
3. слияние:
   - "ce":   порядок только по CE;
   - "rrf":  1/(rrf_k + исходный ранг) + w_ce/(rrf_k + CE-ранг внутри топ-k);
   - "keep": первые m исходных остаются на местах 1..m, места m+1..k — по CE среди остальных.

Запуск (из корня репозитория):
  python experiments/rerank_ce.py --cands work/cands/hybrid_geo2_val.parquet --queries val \
      --model bge-m3 --k 100 --fusion rrf --out work/cands/hybrid_geo2_ce_val.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "work" / "rerank"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Модели-кандидаты (лицензии — по карточкам на Hugging Face)
QWEN_TASK = ("Given a search query on a Russian classifieds site for services, "
             "judge whether the service listing matches the query")
QWEN_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
               "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
               "<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MODELS = {
    # Apache-2.0, 118M (mMiniLMv2, 12 слоёв × 384), обучена на mMARCO (MS MARCO, переведённый на 14 языков, есть ru)
    "minilm": dict(name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", kind="plain", batch=128),
    # Apache-2.0, 568M (XLM-RoBERTa-large из bge-m3), многоязычная
    "bge-m3": dict(name="BAAI/bge-reranker-v2-m3", kind="plain", batch=64),
    # Apache-2.0, 0.6B, Qwen3-Reranker-0.6B, переделанный в Qwen3ForSequenceClassification
    # (логит yes − логит no); промпт-шаблон собираем сами, как в карточке модели
    "qwen3-0.6b": dict(name="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", kind="qwen", batch=32),
}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_ce(key: str, max_len: int = 256, dev: str | None = None):
    """CrossEncoder на GPU в fp16 (на CPU — fp32)."""
    from sentence_transformers import CrossEncoder
    dev = dev or device()
    # на GPU грузим сразу в fp16: загрузка в fp32 и .half() даёт лишний пик VRAM (у bge ×2)
    kw = dict(model_kwargs={"dtype": torch.float16}) if dev == "cuda" else {}
    m = CrossEncoder(MODELS[key]["name"], device=dev, max_length=max_len, **kw)
    if dev == "cuda":
        m.model.half()
    else:
        m.model.float()
    if MODELS[key]["kind"] == "qwen" and m.tokenizer.pad_token is None:
        m.tokenizer.pad_token = m.tokenizer.eos_token
    return m


def load_items(ids=None) -> pd.DataFrame:
    cols = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]
    items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=cols)
    items["item_id"] = items["item_id"].astype(str)
    if ids is not None:
        items = items[items["item_id"].isin(set(ids))]
    return items


def truncate_tokens(tok, texts: list[str], n: int) -> list[str]:
    """Обрезает тексты до n токенов (без спецтокенов) и возвращает строки обратно."""
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=n)["input_ids"]
    return tok.batch_decode(enc, skip_special_tokens=True)


def build_texts(key: str, tok, queries: pd.DataFrame, items: pd.DataFrame, max_len: int = 256):
    """Тексты запросов (query_id -> str) и документов (item_id -> str) в виде, который ждёт модель.

    Запрос = search_query + значения фильтров (как у dense). Документ = item_text, обрезанный так,
    чтобы запрос (≤ 40 токенов) + документ + служебные токены/шаблон уместились в max_len.
    """
    from dense import query_text
    from text import item_text
    qt = dict(zip(queries["query_id"], truncate_tokens(tok, query_text(queries, with_filters=True), 40)))
    kind = MODELS[key]["kind"]
    overhead = 4 if kind == "plain" else 90  # у Qwen длинный промпт-шаблон
    doc_len = max_len - 40 - overhead
    dt = dict(zip(items["item_id"], truncate_tokens(tok, item_text(items).tolist(), doc_len)))
    if kind == "qwen":
        qt = {k: f"{QWEN_PREFIX}<Instruct>: {QWEN_TASK}\n<Query>: {v}\n" for k, v in qt.items()}
        dt = {k: f"<Document>: {v}{QWEN_SUFFIX}" for k, v in dt.items()}
    return qt, dt


def predict_pairs(model, q: list[str], d: list[str], batch: int, chunk: int = 20000, verbose=True) -> np.ndarray:
    """CE-скоры пар. Пары сортируются по длине (меньше паддинга), результат — в исходном порядке."""
    n = len(q)
    order = np.argsort([len(a) + len(b) for a, b in zip(q, d)], kind="stable")
    out = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, chunk):
        idx = order[s:s + chunk]
        sc = model.predict([(q[i], d[i]) for i in idx], batch_size=batch, show_progress_bar=False,
                           convert_to_numpy=True, activation_fn=torch.nn.Identity())
        out[idx] = np.asarray(sc, dtype=np.float32).reshape(-1)
        if verbose:
            done = min(s + chunk, n)
            print(f"  {done}/{n} пар, {done / (time.time() - t0):.0f} пар/с", flush=True)
    return out


def ce_scores(cands: pd.DataFrame, queries: pd.DataFrame, key: str, k: int, items: pd.DataFrame | None = None,
              cache: str | Path | None = None, model=None, max_len: int = 256, verbose=True) -> pd.DataFrame:
    """CE-скоры для пар топ-k кандидатов: DataFrame (query_id, item_id, ce).

    cache — parquet с уже посчитанными скорами этой модели; недостающие пары досчитываются и дописываются.
    """
    top = cands.loc[cands["rank"] <= k, ["query_id", "item_id"]]
    have = pd.DataFrame(columns=["query_id", "item_id", "ce"])
    if cache is not None and Path(cache).exists():
        have = pd.read_parquet(cache)
    need = top.merge(have[["query_id", "item_id"]], how="left", indicator=True)
    need = need[need["_merge"] == "left_only"].drop(columns="_merge")
    if len(need):
        model = model or load_ce(key, max_len)
        if items is None:
            items = load_items(need["item_id"].unique())
        items = items[items["item_id"].isin(set(need["item_id"]))]
        qt, dt = build_texts(key, model.tokenizer, queries[queries["query_id"].isin(set(need["query_id"]))],
                             items, max_len)
        if verbose:
            print(f"{key}: считаем {len(need)} пар", flush=True)
        sc = predict_pairs(model, need["query_id"].map(qt).tolist(), need["item_id"].map(dt).tolist(),
                           MODELS[key]["batch"], verbose=verbose)
        new = need.assign(ce=sc)
        have = new if have.empty else pd.concat([have, new], ignore_index=True)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            have.to_parquet(cache, index=False)
    return top.merge(have, on=["query_id", "item_id"], how="left")


def fuse(cands: pd.DataFrame, ce: pd.DataFrame, k: int, fusion: str = "ce", m: int = 30,
         rrf_k: int = 60, w_ce: float = 1.0) -> pd.DataFrame:
    """Переставляет топ-k по CE-скорам, хвост (rank > k) оставляет как был. Формат выхода = формат входа."""
    c = cands[["query_id", "item_id", "score", "rank"]]
    head = c[c["rank"] <= k].merge(ce[["query_id", "item_id", "ce"]], on=["query_id", "item_id"], how="left")
    assert head["ce"].notna().all(), "не для всех пар топ-k есть CE-скор"
    # CE-ранг внутри топ-k запроса (1 = лучший)
    head["ce_rank"] = head.groupby("query_id")["ce"].rank(ascending=False, method="first")
    if fusion == "ce":
        new = head["ce"].to_numpy(dtype=np.float64)
    elif fusion == "rrf":
        new = 1.0 / (rrf_k + head["rank"].to_numpy()) + w_ce / (rrf_k + head["ce_rank"].to_numpy())
    elif fusion == "keep":
        # первые m исходных — всегда наверху в исходном порядке; остальные — по CE
        keep = head["rank"].to_numpy() <= m
        new = np.where(keep, 1e6 - head["rank"].to_numpy(), head["ce"].to_numpy(dtype=np.float64))
    else:
        raise ValueError(fusion)
    head = head.assign(score=new.astype(np.float32))
    # при равных скорах выше тот, у кого лучше исходный ранг
    head = head.sort_values(["query_id", "score", "rank"], ascending=[True, False, True])
    head["rank"] = head.groupby("query_id").cumcount() + 1
    tail = c[c["rank"] > k]
    out = pd.concat([head[["query_id", "item_id", "score", "rank"]], tail], ignore_index=True)
    out = out.sort_values(["query_id", "rank"], kind="stable").reset_index(drop=True)
    out["rank"] = out["rank"].astype(cands["rank"].dtype)
    return out


def rerank(cands: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame | None = None,
           model_name: str = "bge-m3", k: int = 100, fusion: str = "rrf", m: int = 30, rrf_k: int = 60,
           w_ce: float = 1.0, cache: str | Path | None = None, max_len: int = 256, verbose=True) -> pd.DataFrame:
    """Полный реранк: CE-скоры топ-k + слияние. Возвращает кандидатов в исходном формате (топ-300)."""
    ce = ce_scores(cands, queries, model_name, k, items=items, cache=cache, max_len=max_len, verbose=verbose)
    return fuse(cands, ce, k, fusion, m=m, rrf_k=rrf_k, w_ce=w_ce)


def top_lists(cands: pd.DataFrame, n: int = 50) -> dict:
    """query_id -> первые n item_id (для recall_at_k)."""
    t = cands[cands["rank"] <= n].sort_values(["query_id", "rank"])
    return t.groupby("query_id")["item_id"].agg(list).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--queries", choices=["val", "bench"], required=True)
    ap.add_argument("--model", default="bge-m3", choices=list(MODELS))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--fusion", default="rrf", choices=["ce", "rrf", "keep"])
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--w-ce", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    q = pd.read_parquet(ROOT / "work" / "val_queries.parquet") if a.queries == "val" \
        else pd.read_parquet(DATA / "benchmark_queries.parquet")
    c = pd.read_parquet(a.cands)
    cache = CACHE / f"ce_{a.model}_{a.queries}.parquet"
    t0 = time.time()
    out = rerank(c, q, model_name=a.model, k=a.k, fusion=a.fusion, m=a.m, rrf_k=a.rrf_k, w_ce=a.w_ce, cache=cache)
    out.to_parquet(a.out, index=False)
    print(f"готово за {time.time() - t0:.0f} с -> {a.out}")


if __name__ == "__main__":
    main()
