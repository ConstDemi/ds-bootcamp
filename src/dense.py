"""Dense-ретривер: эмбеддинги запросов и объявлений + поиск по косинусу.

Схема:
1. документ = item_text(items) (заголовок + очищенные параметры + начало описания) с префиксом модели;
2. запрос = search_query (опционально + значения фильтров) с префиксом модели;
3. эмбеддинги нормируются, косинус = скалярное произведение; поиск — матричное умножение батчами
   (на GPU, если есть, иначе на CPU; 189k × 384 во float32 — ~0.3 ГБ, FAISS не нужен);
4. два режима: all — весь корпус; geo — для городских запросов сначала объявления того же города,
   добор до k лучшими из всего корпуса; для регионов и «всей России» — весь корпус.

Эмбеддинги корпуса кешируются в work/emb/<ключ модели>.npy (float16) + <ключ>_ids.npy (порядок item_id)
+ <ключ>.json (параметры кодирования). Кодирование идёт кусками с сохранением на диск,
поэтому прерванный запуск продолжается с места остановки.

Запуск кодирования корпуса:  python src/dense.py encode e5-small --max-seq 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB = ROOT / "work" / "emb"

# Модели-кандидаты. Префиксы взяты из карточек моделей на Hugging Face.
MODELS = {
    # MIT, 118M параметров (из них большая часть — словарь), размерность 384
    "e5-small": dict(name="intfloat/multilingual-e5-small", q="query: ", d="passage: "),
    # MIT, 278M, размерность 768
    "e5-base": dict(name="intfloat/multilingual-e5-base", q="query: ", d="passage: "),
    # Apache-2.0, 124M, размерность 768, русская модель (дообучена из deepvk/deberta-v1-base)
    # sentence-transformers 6.1 не читает её modules.json (Normalize с path="" падает на config.json
    # трансформера), поэтому модули собираем вручную: трансформер + mean-pooling + нормировка (как в 1_Pooling)
    "user-base": dict(name="deepvk/USER-base", q="query: ", d="passage: ", manual_pooling="mean"),
    # Apache-2.0, 0.6B, размерность 1024; запрос с инструкцией, документ без префикса
    "qwen3-emb-0.6b": dict(name="Qwen/Qwen3-Embedding-0.6B", d="",
                           q="Instruct: Given a user search query on a Russian classifieds site for services, "
                             "retrieve service listings that match the query\nQuery:"),
    # MIT, 568M (XLM-RoBERTa-large), размерность 1024; dense-вектор (CLS), префиксы не нужны (карточка модели)
    "bge-m3": dict(name="BAAI/bge-m3", q="", d=""),
}

ITEM_COLS = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw", "item_location_id"]


def device() -> str:
    """Устройство для инференса: GPU, если есть."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(key: str, max_seq: int = 256, quantize: bool | None = None):
    """Загружает модель sentence-transformers.

    На GPU — fp16. На CPU по умолчанию int8-квантизация линейных слоёв (quantize_dynamic):
    в ~1.6 раза быстрее fp32, но на подкорпусе теряет ~0.03 Recall@50 (см. eda/07_dense.md),
    поэтому это вынужденный режим «нет GPU».
    """
    from sentence_transformers import SentenceTransformer
    dev = device()
    spec = MODELS[key]
    if "manual_pooling" in spec:
        from sentence_transformers import models as stm
        tr = stm.Transformer(spec["name"], max_seq_length=max_seq)
        pool = stm.Pooling(tr.get_word_embedding_dimension(), spec["manual_pooling"])
        m = SentenceTransformer(modules=[tr, pool, stm.Normalize()], device=dev)
    else:
        m = SentenceTransformer(spec["name"], device=dev)
    if dev == "cuda":
        m = m.half()
    else:
        m = m.float()  # часть моделей (Qwen3) хранится в bf16 — на CPU считаем в fp32
    if dev == "cpu" and (quantize is None or quantize):
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    m.max_seq_length = max_seq
    return m


def query_text(queries: pd.DataFrame, with_filters: bool = False) -> list[str]:
    """Текст запроса для модели. with_filters=True дописывает значения фильтров
    («Вид услуги Ремонт и отделка» → «Ремонт и отделка»), они уточняют смысл коротких запросов."""
    text = queries["search_query"].fillna("").astype(str)
    if with_filters:
        from text import clean_params  # тот же очиститель служебных ключей, что и для объявлений
        f = clean_params(queries["search_infm_params_text"])
        text = text.where(f.eq(""), text + " (" + f + ")")
    return text.tolist()


def encode_queries(model, key: str, queries: pd.DataFrame, with_filters: bool = False,
                   batch_size: int = 128) -> np.ndarray:
    """Нормированные эмбеддинги запросов (float32), порядок строк как в queries."""
    texts = [MODELS[key]["q"] + t for t in query_text(queries, with_filters)]
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, prompt="",
                        convert_to_numpy=True).astype(np.float32)


def encode_corpus(key: str, max_seq: int = 256, batch_size: int = 64, chunk: int = 8192,
                  items: pd.DataFrame | None = None, tag: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Кодирует корпус и сохраняет work/emb/<tag>.npy (float16) и <tag>_ids.npy.

    Куски по chunk документов пишутся в work/emb/<tag>_parts/ — при перезапуске готовые куски
    пропускаются. Возвращает (эмбеддинги [n, dim] float16, item_id [n]).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from text import item_text

    tag = tag or key
    EMB.mkdir(parents=True, exist_ok=True)
    if items is None:
        items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS[:-1])
    docs = (MODELS[key]["d"] + item_text(items)).tolist()
    ids = items["item_id"].astype(str).to_numpy()
    parts = EMB / f"{tag}_parts"
    parts.mkdir(exist_ok=True)

    model = load_model(key, max_seq)
    t0 = time.time()
    n_chunks = (len(docs) + chunk - 1) // chunk
    for c in range(n_chunks):
        f = parts / f"{c:04d}.npy"
        if f.exists():
            continue
        e = model.encode(docs[c * chunk:(c + 1) * chunk], batch_size=batch_size, prompt="",
                         normalize_embeddings=True, convert_to_numpy=True)
        np.save(f, e.astype(np.float16))
        done = min((c + 1) * chunk, len(docs))
        print(f"[{tag}] {done}/{len(docs)} за {time.time() - t0:.0f} с", flush=True)
    emb = np.concatenate([np.load(parts / f"{c:04d}.npy") for c in range(n_chunks)])
    np.save(EMB / f"{tag}.npy", emb)
    np.save(EMB / f"{tag}_ids.npy", ids)
    meta = dict(model=MODELS[key]["name"], max_seq=max_seq, device=device(), batch=batch_size,
                gpu=torch.cuda.get_device_name(0) if device() == "cuda" else None,
                q_prefix=MODELS[key]["q"], d_prefix=MODELS[key]["d"],
                precision="fp16" if device() == "cuda" else "int8-dynamic (CPU)",
                n=len(ids), dim=int(emb.shape[1]), seconds_this_run=round(time.time() - t0, 1))
    (EMB / f"{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return emb, ids


def load_corpus_emb(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Кешированные эмбеддинги корпуса и порядок item_id."""
    return np.load(EMB / f"{tag}.npy"), np.load(EMB / f"{tag}_ids.npy", allow_pickle=True)


def _topk(q: torch.Tensor, d: torch.Tensor, k: int, bs: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Топ-k по косинусу для каждой строки q среди строк d (обе нормированы). Батчи по bs запросов."""
    k = min(k, d.shape[0])
    idx, sc = [], []
    for s in range(0, q.shape[0], bs):
        v, i = torch.topk(q[s:s + bs] @ d.T, k, dim=1)
        idx.append(i.cpu().numpy()); sc.append(v.float().cpu().numpy())
    return np.concatenate(idx), np.concatenate(sc)


def search(q_emb: np.ndarray, d_emb: np.ndarray, queries: pd.DataFrame, item_locs: np.ndarray,
           regions: set, k: int = 300, mode: str = "all") -> list[list[tuple[int, float]]]:
    """Поиск: для каждого запроса список (индекс документа, косинус) длины k, без повторов.

    mode="all" — весь корпус. mode="geo" — если search_location_id город (не в regions),
    сначала топ среди объявлений этого города, потом добор лучшими из всего корпуса.
    """
    dev = device()
    dt = torch.float16 if dev == "cuda" else torch.float32
    D = torch.from_numpy(d_emb).to(dev, dt)
    Q = torch.from_numpy(q_emb).to(dev, dt)
    all_i, all_s = _topk(Q, D, k)
    res = [list(zip(i.tolist(), s.tolist())) for i, s in zip(all_i, all_s)]
    if mode == "all":
        return res
    assert mode == "geo"
    locs = queries["search_location_id"].to_numpy()
    # Индекс объявлений по городу: location_id -> массив позиций в корпусе.
    order = np.argsort(item_locs, kind="stable")
    uniq, starts = np.unique(item_locs[order], return_index=True)
    pools = dict(zip(uniq.tolist(), np.split(order, starts[1:])))
    for loc in np.unique(locs):
        if loc in regions or loc not in pools:
            continue  # регион / вся Россия / город без объявлений — весь корпус
        rows = np.flatnonzero(locs == loc)
        pool = pools[loc]
        pi, ps = _topk(Q[rows], D[torch.from_numpy(pool).to(dev)], k)
        for r, i_row, s_row in zip(rows, pi, ps):
            city = [(int(pool[i]), float(s)) for i, s in zip(i_row, s_row)]
            seen = {i for i, _ in city}
            fill = [(i, s) for i, s in res[r] if i not in seen]
            res[r] = (city + fill)[:k]
    return res


def to_frame(res: list, queries: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Кандидаты в длинном формате: query_id, item_id (str), score (float), rank (с 1)."""
    qid, iid, sc, rk = [], [], [], []
    for q, lst in zip(queries["query_id"], res):
        for r, (i, s) in enumerate(lst, 1):
            qid.append(q); iid.append(str(ids[i])); sc.append(s); rk.append(r)
    return pd.DataFrame({"query_id": qid, "item_id": iid, "score": np.array(sc, dtype=np.float32),
                         "rank": np.array(rk, dtype=np.int32)})


def to_preds(res: list, queries: pd.DataFrame, ids: np.ndarray, k: int = 50) -> dict:
    """query_id -> список из k item_id (строки) для recall_at_k и answer.csv."""
    return {q: [str(ids[i]) for i, _ in lst[:k]] for q, lst in zip(queries["query_id"], res)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode"])
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    encode_corpus(a.model, a.max_seq, a.batch, tag=a.tag)
