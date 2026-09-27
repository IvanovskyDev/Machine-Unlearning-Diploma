#!/usr/bin/env bash
# Самопроверка окружений unl и atk (план, блок 9): по каждой проверке печатает OK или FAIL.
# Запускать после любых изменений окружений и на новой машине. В Colab:
#   bash /content/repo/scripts/check_env.sh
# Где лежат репозиторий и окружения, можно поменять переменными REPO, UNL и ATK.

REPO=${REPO:-/content/repo}   # ${X:-y} — значение переменной X, а если её нет, то y
UNL=${UNL:-/content/envs/unl}
ATK=${ATK:-/content/envs/atk}
unset PYTHONPATH MPLBACKEND   # настройки Colab, которые мешают нашим окружениям (разбор D)

LOG=$(mktemp)   # временный файл: сюда пишется вывод каждой проверки
FAILS=0         # сколько проверок не прошло

# check "название" команда … — выполняет команду; успех — OK, ошибка — FAIL и конец вывода
check() {
  local name=$1
  shift   # дальше в "$@" остаётся только команда
  if "$@" >"$LOG" 2>&1; then
    echo "OK    $name"
  else
    echo "FAIL  $name"
    tail -n 5 "$LOG" | sed 's/^/        /'   # последние строки вывода: по ним видна причина
    FAILS=$((FAILS + 1))
  fi
}

# пакеты окружения ровно те, что в lock-файле; lock-файлы записаны так же в части B
unl_matches_lock() {
  uv pip freeze --python "$UNL/bin/python" | grep -v -e flash-attn -e open-unlearning |
    diff - "$REPO/envs/requirements-unl.lock"
}
atk_matches_lock() {
  # --exclude-editable: пакет urec стоит из папки репозитория, в lock-файле его нет
  uv pip freeze --python "$ATK/bin/python" --exclude-editable |
    diff - "$REPO/envs/requirements-atk.lock"
}

echo "Окружение unl: $UNL"
check "пакеты совпадают с envs/requirements-unl.lock" unl_matches_lock
check "torch видит GPU, FlashAttention импортируется" "$UNL/bin/python" -c "
import torch, flash_attn
assert torch.cuda.is_available(), 'torch не видит GPU'
"

echo "Окружение atk: $ATK"
check "пакеты совпадают с envs/requirements-atk.lock" atk_matches_lock
check "torch видит GPU, vLLM импортируется" "$ATK/bin/python" -c "
import torch, vllm
assert torch.cuda.is_available(), 'torch не видит GPU'
"
check "spaCy на GPU находит имя человека" "$ATK/bin/python" -c "
import spacy
assert spacy.prefer_gpu(), 'spaCy не видит GPU'
doc = spacy.load('en_core_web_trf')('Basil Mahfouz Al-Kuwaiti was born in Kuwait City in 1956.')
assert 'PERSON' in [entity.label_ for entity in doc.ents], doc.ents
"
check "BERTScore считает F1" "$ATK/bin/python" -c "
import math
from bert_score import BERTScorer
scorer = BERTScorer(lang='en', model_type='roberta-large', rescale_with_baseline=True)
f1 = scorer.score(['He was born in Kuwait.'], ['The author was born in Kuwait City.'])[2].item()
assert math.isfinite(f1), f1
"
check "пакет urec установлен" "$ATK/bin/python" -c "import urec"
check "тесты urec без GPU и интернета проходят" "$ATK/bin/python" -m pytest -q -p no:cacheprovider \
  -m "not gpu and not network" "$REPO/tests"

rm -f "$LOG"
echo
if [ "$FAILS" -eq 0 ]; then
  echo "OK: все проверки прошли"
else
  echo "FAIL: не прошли проверки: $FAILS"
  exit 1
fi
