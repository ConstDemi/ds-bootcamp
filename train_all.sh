#!/usr/bin/env bash
# Обучение итоговой схемы с нуля (сабмит 5): всё учится только на train[train_mask], валидация не участвует.
# Запуск из корня репозитория: bash train_all.sh   (или PY=/путь/к/python bash train_all.sh)
# Требования, время шагов и ожидаемый результат — README.md, раздел «Как воспроизвести».
# Повторный запуск продолжает с места обрыва: готовые модели и эмбеддинги корпуса пропускаются.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
FT=(--model BAAI/bge-m3 --q-prefix "" --d-prefix "" --lora 16 --lr 1e-4 --grad-ckpt --batch 128 --max-pairs 0 --pretok)
A=(--out work/ltr_ft_all --fold-model "bge-m3-lora-all-f{f}" --val-model bge-m3-lora-all)

step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "1. валидация и train_mask"
$PY src/validation.py

step "2. обучающий пул (21 439 запросов, 5 фолдов по тексту) и кандидаты гео-схемы; корпус bge-m3 кодируется при первом запуске"
$PY src/ltr_big.py sample
$PY src/ltr_big.py retrieve

step "3. дообучение bge-m3 (LoRA): общая модель и 5 копий без своего фолда"
train() {   # $1 = имя модели, $2.. = доп. флаги
  local name=$1; shift
  if [ -f "work/models/$name/train_info.json" ]; then echo "$name: уже обучена"; return; fi
  $PY src/ft_train.py "${FT[@]}" --name "$name" "$@"
}
train bge-m3-lora-all
for f in 0 1 2 3 4; do train "bge-m3-lora-all-f$f" --excl-fold $f; done

step "4. эмбеддинги корпуса дообученными моделями"
for m in bge-m3-lora-all bge-m3-lora-all-f0 bge-m3-lora-all-f1 bge-m3-lora-all-f2 bge-m3-lora-all-f3 bge-m3-lora-all-f4; do
  if [ -f "work/emb/${m}_s256.npy" ]; then echo "$m: корпус уже закодирован"; continue; fi
  $PY src/ft_eval.py encode --name "$m" --batch 64
done

step "5. кандидаты, признаки, проверка утечки и 5 бустеров LightGBM"
$PY src/ltr_ft.py retrieve "${A[@]}"
$PY src/ltr_ft.py features "${A[@]}"
$PY src/ltr_ft.py leak "${A[@]}"
$PY src/ltr_ft.py final "${A[@]}" --variants main

step "готово: python src/pipeline.py --mode val   (ожидается Recall@50 около 0.915)"
