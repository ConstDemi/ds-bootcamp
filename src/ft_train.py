"""Дообучение модели эмбеддингов на парах «запрос → выбранное объявление» из train.

Схема:
1. Пары берутся из строк train[train_mask] (отложенные запросы валидации в обучение не попадают;
   для бенчмарка можно переобучить на полном train флагом --full-train).
   Запрос = search_query + значения фильтров (как dense.query_text(with_filters=True)),
   документ = item_text() выбранного объявления из той же строки (item_* колонки train).
2. Дубли пар (запрос с фильтрами, item_id) убираются. На один нормализованный текст запроса берётся
   не больше --cap пар (случайных, seed), чтобы «маникюр» (6 тыс. пар) не заслонил хвост.
3. Лосс — MultipleNegativesRankingLoss (in-batch negatives): косинус × scale 20, кросс-энтропия,
   правильный документ — свой, негативы — документы остальных пар батча.
4. Батчи без дублей (аналог NoDuplicatesBatchSampler): в одном батче не бывает одинаковых
   нормализованных текстов запроса, одинаковых item_id и одинаковых текстов документа — иначе
   «негатив» был бы на самом деле правильным ответом.
5. Цикл обучения написан вручную на torch (AdamW, линейный warmup + спад, fp16 autocast + GradScaler):
   SentenceTransformerTrainer требует пакеты datasets и accelerate, которых нет в окружении.
   Модель — обычный SentenceTransformer, веса сохраняются model.save() и читаются SentenceTransformer(path).

Запуск из корня репозитория:
    python src/ft_train.py --model intfloat/multilingual-e5-small --name e5-small-ft --batch 256
    python src/ft_train.py ... --max-pairs 2000 --name smoke      # быстрая проверка
Выход: work/models/<name>/ (веса) + work/models/<name>/train_info.json (параметры, время, лосс).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text import clean_params, item_text  # noqa: E402
from validation import load_validation, normalize_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS_DIR = ROOT / "work" / "models"

TRAIN_COLS = ["search_query", "search_infm_params_text", "item_id",
              "item_title_raw", "item_infm_params_text", "item_description_raw"]


def build_pairs(full_train: bool = False, cap: int = 20, seed: int = 0, excl_fold: int = -1) -> pd.DataFrame:
    """Пары для обучения: q (текст запроса для модели), d (текст документа), tn (норм. текст запроса),
    item_id, dh (хеш текста документа). Без префиксов модели.
    excl_fold >= 0: перекрёстный режим для LTR-big — из train[mask] убираются строки с
    work/ltr_big/excl_fold.npy == excl_fold (запросы пула фолда f и «новые» тексты фолда f),
    т.е. модель учится ровно на train_fit фолда f из src/ltr_big.py."""
    import pyarrow.parquet as pq
    # сначала только лёгкие колонки запроса; тексты объявлений читаем потом для отобранных строк (RAM)
    tr = pd.read_parquet(DATA / "train.parquet", columns=TRAIN_COLS[:3])
    if not full_train:
        _, _, mask = load_validation()
        if excl_fold >= 0:
            excl = np.load(ROOT / "work" / "ltr_big" / "excl_fold.npy")
            assert len(excl) == len(mask)
            mask = mask & (excl != excl_fold)
        tr = tr[mask]
    # запрос как в dense.query_text(with_filters=True)
    text = tr["search_query"].fillna("").astype(str)
    f = clean_params(tr["search_infm_params_text"])
    q = text.where(f.eq(""), text + " (" + f + ")")
    p = pd.DataFrame({"q": q, "tn": normalize_text(tr["search_query"]),
                      "qkey": normalize_text(q), "item_id": tr["item_id"].astype(str)})  # индекс = строка train
    del tr, text, f, q
    n_rows = len(p)
    # 1) дубли пар (запрос с фильтрами, item_id)
    p = p[~p.duplicated(["qkey", "item_id"])]
    n_dedup = len(p)
    # 2) не больше cap пар на нормализованный текст запроса (случайные, воспроизводимо)
    rng = np.random.default_rng(seed)
    p = p.assign(_r=rng.random(len(p))).sort_values(["tn", "_r"])
    p = p[p.groupby("tn").cumcount() < cap].drop(columns=["_r", "qkey"]).sort_index()
    # 3) документ — только для оставшихся строк; train читаем кусками, чтобы не держать все описания в RAM
    need = p.index.to_numpy()
    docs, start = [], 0
    for rb in pq.ParquetFile(DATA / "train.parquet").iter_batches(batch_size=50_000, columns=TRAIN_COLS[3:]):
        sel = need[(need >= start) & (need < start + rb.num_rows)] - start
        docs.append(item_text(rb.take(sel).to_pandas()).to_numpy())
        start += rb.num_rows
    p["d"] = np.concatenate(docs)
    del docs
    p["dh"] = pd.util.hash_pandas_object(p["d"], index=False).to_numpy()
    p = p.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    p = p[["q", "d", "tn", "item_id", "dh"]]
    p.attrs.update(n_rows=n_rows, n_dedup=n_dedup, n_capped=len(p))
    return p


def no_dup_batches(p: pd.DataFrame, batch: int, seed: int = 0) -> list[np.ndarray]:
    """Батчи без повторов текста запроса / item_id / текста документа внутри батча.
    Жадно, как NoDuplicatesBatchSampler: идём по перемешанному списку, конфликтующие пары
    откладываются в следующий батч. Неполный последний батч выбрасывается."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(p))
    tn = pd.factorize(p["tn"])[0]
    it = pd.factorize(p["item_id"])[0]
    dh = pd.factorize(p["dh"])[0]
    batches, rest = [], list(order)
    while len(rest) >= batch:
        cur, s_t, s_i, s_d, left = [], set(), set(), set(), []
        stop = len(rest)
        for pos, j in enumerate(rest):
            if len(cur) == batch:
                stop = pos
                break
            if tn[j] not in s_t and it[j] not in s_i and dh[j] not in s_d:
                cur.append(j); s_t.add(tn[j]); s_i.add(it[j]); s_d.add(dh[j])
            else:
                left.append(j)
        if len(cur) < batch:
            break
        batches.append(np.array(cur))
        rest = left + rest[stop:]
    return batches


def embed(model, texts: list[str], max_len: int, dev: str) -> torch.Tensor:
    """Эмбеддинги с графом градиентов (pooling и нормировка — модули самого SentenceTransformer)."""
    feats = model.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    feats = {k: v.to(dev) for k, v in feats.items()}
    out = model(feats)["sentence_embedding"]
    return F.normalize(out, dim=-1)


def pretokenize(model, texts: list[str], max_len: int) -> list[list[int]]:
    """Токенизация всех текстов заранее, одним пакетом (быстрый токенизатор параллелит на Rust).
    На машине с медленным CPU токенизация в цикле обучения заметно тормозит шаг."""
    return model.tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]


def embed_ids(model, ids: list[list[int]], dev: str) -> torch.Tensor:
    """Как embed, но из готовых id токенов: только паддинг внутри батча."""
    feats = model.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    feats = {k: v.to(dev) for k, v in feats.items()}
    out = model(feats)["sentence_embedding"]
    return F.normalize(out, dim=-1)


def main():
    """Дообучение эмбеддера MNRL на парах «запрос -> выбранное объявление» из train[mask] (см. docstring модуля)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--name", required=True)
    ap.add_argument("--q-prefix", default="query: ")
    ap.add_argument("--d-prefix", default="passage: ")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--q-len", type=int, default=64)
    ap.add_argument("--d-len", type=int, default=128)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = все пары после cap")
    ap.add_argument("--full-train", action="store_true", help="пары из полного train (только для финала)")
    ap.add_argument("--excl-fold", type=int, default=-1,
                    help="перекрёстный режим: без строк train с excl_fold.npy == f (-1 = прежний режим)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--lora", type=int, default=0, help="ранг LoRA (0 = полный fine-tune); нужен peft")
    ap.add_argument("--freeze-emb", action="store_true", help="заморозить матрицу словаря")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretok", action="store_true", help="токенизировать все пары до обучения (быстрее на слабом CPU)")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(a.seed)
    dev = "cuda"
    t0 = time.time()
    if a.excl_fold >= 0:
        assert not a.full_train, "--excl-fold только вместе с train[mask]"
    p = build_pairs(a.full_train, a.cap, a.seed, a.excl_fold)
    info_pairs = dict(p.attrs)
    if a.max_pairs:
        p = p.iloc[:a.max_pairs].reset_index(drop=True)
    batches = no_dup_batches(p, a.batch, a.seed)
    t_data = time.time() - t0
    print(f"пары: строк {info_pairs['n_rows']}, без дублей {info_pairs['n_dedup']}, после cap={a.cap} "
          f"{info_pairs['n_capped']}, в обучении {len(p)}; батчей {len(batches)} по {a.batch}; "
          f"уникальных текстов {p.tn.nunique()}; подготовка {t_data:.0f} с", flush=True)

    model = SentenceTransformer(a.model, device=dev)
    # полный fine-tune: мастер-веса fp32 (+ autocast fp16); LoRA: замороженная основа в fp16 (экономит ~1 ГБ
    # у bge-m3), адаптеры peft сам держит в fp32 (autocast_adapter_dtype)
    model.half() if a.lora else model.float()
    tr_mod = model[0]
    if a.grad_ckpt:
        # нереентерабельный вариант: с LoRA входы слоёв не требуют градиента, реентерабельный их бы потерял
        tr_mod.auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.freeze_emb:
        tr_mod.auto_model.embeddings.word_embeddings.weight.requires_grad_(False)
    if a.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05,
                         target_modules=["query", "key", "value", "dense"])
        # в sentence-transformers 6 auto_model — свойство только для чтения (возвращает .model),
        # поэтому обёртку peft кладём в атрибут model
        tr_mod.model = get_peft_model(tr_mod.model, cfg)
        tr_mod.model.print_trainable_parameters()
    params = [x for x in model.parameters() if x.requires_grad]
    n_train = sum(x.numel() for x in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    total = len(batches) * a.epochs
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda")
    labels = torch.arange(a.batch, device=dev)
    qs = (a.q_prefix + p["q"]).tolist()
    ds = (a.d_prefix + p["d"]).tolist()
    if a.pretok:
        t2 = time.time()
        q_ids, d_ids = pretokenize(model, qs, a.q_len), pretokenize(model, ds, a.d_len)
        print(f"предтокенизация {len(qs)} пар за {time.time() - t2:.0f} с", flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    losses, step = [], 0
    for ep in range(a.epochs):
        ep_batches = batches if ep == 0 else no_dup_batches(p, a.batch, a.seed + ep)
        for b in ep_batches:
            with torch.autocast("cuda", dtype=torch.float16):
                if a.pretok:
                    qe = embed_ids(model, [q_ids[j] for j in b], dev)
                    de = embed_ids(model, [d_ids[j] for j in b], dev)
                else:
                    qe = embed(model, [qs[j] for j in b], a.q_len, dev)
                    de = embed(model, [ds[j] for j in b], a.d_len, dev)
                logits = a.scale * qe.float() @ de.float().T
                loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0 or step == total:
                el = time.time() - t1
                print(f"  шаг {step}/{total} лосс {np.mean(losses[-50:]):.4f} "
                      f"{el:.0f} с (≈{el / step * total / 60:.1f} мин всего), "
                      f"пик GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} ГБ", flush=True)
    t_train = time.time() - t1

    out = MODELS_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    if a.lora:
        tr_mod.model = tr_mod.model.merge_and_unload()  # LoRA вливается в веса -> обычная модель
    model.save(str(out))
    info = dict(base_model=a.model, args=vars(a), pairs=info_pairs, pairs_used=len(p),
                unique_texts=int(p.tn.nunique()), batches=len(batches), steps=step,
                trainable_params=int(n_train), data_seconds=round(t_data, 1), train_seconds=round(t_train, 1),
                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                gpu=torch.cuda.get_device_name(0), loss_first50=float(np.mean(losses[:50])),
                loss_last50=float(np.mean(losses[-50:])),
                loss_curve=[round(float(np.mean(losses[i:i + 50])), 4) for i in range(0, len(losses), 50)])
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"готово: обучение {t_train / 60:.1f} мин, сохранено в {out}", flush=True)


if __name__ == "__main__":
    main()
