"""Проверка answer.csv перед отправкой: формат и id, но не качество.

Запуск из корня репозитория:
    python check_answer.py answer.csv --data data

Ловит и то, что платформа отклонит (колонки, лишние/недостающие query_id, повторы, >50),
и то, что она молча не засчитает (item_id, которых нет в корпусе: регистр, пробелы, числа).
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ID_RE = re.compile(r"^[0-9a-f]{16}$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answer", nargs="?", default="answer.csv")
    ap.add_argument("--data", default="data", help="папка с benchmark_*.parquet")
    args = ap.parse_args()

    errors, warnings = [], []
    raw = Path(args.answer).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("файл начинается с BOM: сохраняйте в utf-8, а не utf-8-sig")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"FAIL: файл не в UTF-8 ({e})")
        return 1
    header = text.splitlines()[0] if text else ""
    if header.lstrip("﻿") != "query_id,answer":
        errors.append(f"первая строка должна быть ровно 'query_id,answer', а она {header!r}")

    # dtype=str и keep_default_na=False: id остаются строками, ведущие нули не теряются
    ans = pd.read_csv(args.answer, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    queries = pd.read_parquet(Path(args.data) / "benchmark_queries.parquet", columns=["query_id"])
    items = pd.read_parquet(Path(args.data) / "benchmark_items.parquet", columns=["item_id"])
    need_q = set(queries["query_id"].astype(str))
    corpus = set(items["item_id"].astype(str))

    if list(ans.columns) != ["query_id", "answer"]:
        errors.append(f"колонки должны быть ['query_id', 'answer'], а они {list(ans.columns)}")
        return report(errors, warnings)

    dup_q = ans["query_id"][ans["query_id"].duplicated()]
    if len(dup_q):
        errors.append(f"повторы query_id: {len(dup_q)}, например {dup_q.iloc[0]!r}")
    got_q = set(ans["query_id"])
    if need_q - got_q:
        errors.append(f"нет строк для {len(need_q - got_q)} query_id, например {sorted(need_q - got_q)[0]!r}")
    if got_q - need_q:
        errors.append(f"лишние query_id: {len(got_q - need_q)}, например {sorted(got_q - need_q)[0]!r}")

    sizes, bad_fmt, unknown, too_many, dups, spacing = [], 0, 0, 0, 0, 0
    example_unknown = None
    for s in ans["answer"]:
        if s != " ".join(s.split()) or "," in s or "[" in s:
            spacing += 1  # двойные пробелы, пробел по краям, запятые, скобки списка
        ids = s.split()
        sizes.append(len(set(ids)))
        too_many += len(ids) > 50
        dups += len(ids) != len(set(ids))
        for i in ids:
            if not ID_RE.match(i):
                bad_fmt += 1
            if i not in corpus:
                unknown += 1
                example_unknown = example_unknown or i

    if spacing:
        errors.append(f"строк с неправильными разделителями (нужен ровно один пробел): {spacing}")
    if too_many:
        errors.append(f"строк с больше чем 50 id: {too_many}")
    if dups:
        errors.append(f"строк с повторами id внутри: {dups}")
    if bad_fmt:
        warnings.append(f"id не в формате 16 символов 0-9a-f: {bad_fmt}")
    if unknown:
        warnings.append(f"id нет в корпусе, они молча не засчитаются: {unknown}, например {example_unknown!r}")

    sz = pd.Series(sizes)
    short = int((sz < 50).sum())
    if short:
        warnings.append(f"строк меньше чем с 50 id: {short} (пустые места не приносят recall)")
    print(f"строк {len(ans)}, запросов в бенчмарке {len(need_q)}, "
          f"id в строке: мин {sz.min()}, среднее {sz.mean():.1f}, макс {sz.max()}")
    return report(errors, warnings)


def report(errors, warnings) -> int:
    for w in warnings:
        print("WARN:", w)
    for e in errors:
        print("FAIL:", e)
    if errors:
        print("Итог: ОТПРАВЛЯТЬ НЕЛЬЗЯ")
        return 1
    print("Итог: формат в порядке" + (", но посмотрите WARN" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
