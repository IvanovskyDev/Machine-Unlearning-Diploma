"""Поддельный src/eval.py форка для тестов оркестратора: вместо оценки пишет файлы TOFU.

Это не тест, а помощник. В TOFU_EVAL.json он записывает добавки, с которыми его
запустили: так тест видит, какую модель и какие сплиты передал оркестратор.
Поведение задают переменные окружения:
- FAKE_FORK_FAIL=eval — упасть, как упавшая оценка;
- FAKE_FORK_EMPTY=eval — завершиться без ошибки, но ничего не записать.
"""

import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]  # --config-name=… и добавки «ключ=значение»
print("fake eval:", " ".join(args), flush=True)  # попадёт в лог шага

if os.environ.get("FAKE_FORK_FAIL") == "eval":
    sys.exit("fake eval: ошибка оценки")  # текст — в поток ошибок, код завершения 1
if os.environ.get("FAKE_FORK_EMPTY") == "eval":
    sys.exit(0)

overrides = dict(arg.split("=", 1) for arg in args if not arg.startswith("--"))
output = Path(overrides["paths.output_dir"])
(output / ".hydra").mkdir(parents=True, exist_ok=True)
(output / ".hydra" / "config.yaml").write_text("\n".join(args) + "\n", encoding="utf-8")
(output / "TOFU_EVAL.json").write_text(json.dumps(overrides), encoding="utf-8")
summary = {"forget_quality": 0.5, "model_utility": 0.6}
(output / "TOFU_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
