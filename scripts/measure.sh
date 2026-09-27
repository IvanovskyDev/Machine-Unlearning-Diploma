#!/usr/bin/env bash
# Запускает команду и замеряет время работы и пиковую память GPU.
# Как вызывать: bash /content/repo/scripts/measure.sh <имя запуска> <команда…>
# Итог дописывается строкой в $BIG/logs/runs.tsv: дата, имя, секунды, пик памяти (МиБ), код завершения.

name=$1        # первое слово после имени скрипта — имя запуска
shift          # убрать его: остальные слова — сама команда

# каждую секунду записывать занятую память GPU в файл; & — в фоне, параллельно с командой
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > /content/gpu_memory.log &
monitor=$!     # номер фонового процесса, чтобы потом его остановить

start=$(date +%s)   # время старта в секундах
"$@"                # выполнить команду
status=$?           # код завершения команды: 0 — успешно
end=$(date +%s)
kill $monitor       # остановить замер памяти

seconds=$((end - start))
peak=$(sort -n /content/gpu_memory.log | tail -n 1)   # самое большое значение из записанных
echo "$name: $((seconds / 60)) мин $((seconds % 60)) с, пик памяти GPU $peak МиБ, код завершения $status"
echo -e "$(date +%F)\t$name\t$seconds\t$peak\t$status" >> $BIG/logs/runs.tsv
exit $status
