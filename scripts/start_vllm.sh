#!/usr/bin/env bash
# Запускает сервер vLLM с моделью из указанной папки и ждёт, пока он будет готов.
# Как вызывать: bash /content/repo/scripts/start_vllm.sh <папка модели>
source /content/envs/atk/bin/activate

# nohup … & — сервер работает в фоне; всё, что он пишет, идёт в лог на Drive;
# --dtype bfloat16 — считать в bf16, даже если веса сохранены во float32 (модели шагов 12–13)
nohup vllm serve "$1" --served-model-name victim --port 8000 \
    --dtype bfloat16 --gpu-memory-utilization 0.4 --max-model-len 4096 \
    --generation-config vllm --seed 0 > $BIG/logs/vllm.log 2>&1 &

# ждать до 10 минут: каждые 5 секунд спрашивать сервер, готов ли он
for i in $(seq 1 120); do
    sleep 5
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "Сервер готов: $1"
        exit 0
    fi
done
echo "Сервер не запустился за 10 минут. Последние строки лога:"
tail -n 30 $BIG/logs/vllm.log
exit 1
