#!/usr/bin/env bash
# Останавливает сервер vLLM и показывает, что память GPU освободилась.
pkill -f "vllm serve"                  # остановить сервер
sleep 15                               # дать ему время завершиться
pkill -f "VLLM::EngineCore"            # на всякий случай — его вычислительный процесс
nvidia-smi --query-gpu=memory.used --format=csv   # занятая память GPU: должна быть почти 0
