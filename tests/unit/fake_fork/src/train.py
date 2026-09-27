"""Поддельный src/train.py форка для тестов оркестратора: вместо обучения пишет «чекпоинт».

Это не тест, а помощник. Как настоящий, он получает добавки Hydra «ключ=значение»
и пишет результат в paths.output_dir. Поведение задают переменные окружения:
- FAKE_FORK_FAIL=train — упасть, как упавшее обучение;
- FAKE_FORK_EMPTY=train — завершиться без ошибки, но ничего не записать.
"""

import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]  # --config-name=… и добавки «ключ=значение»
print("fake train:", " ".join(args), flush=True)  # попадёт в лог шага

if os.environ.get("FAKE_FORK_FAIL") == "train":
    sys.exit("fake train: ошибка обучения")  # текст — в поток ошибок, код завершения 1
if os.environ.get("FAKE_FORK_EMPTY") == "train":
    sys.exit(0)

overrides = dict(arg.split("=", 1) for arg in args if not arg.startswith("--"))
output = Path(overrides["paths.output_dir"])
(output / ".hydra").mkdir(parents=True, exist_ok=True)
(output / ".hydra" / "config.yaml").write_text("\n".join(args) + "\n", encoding="utf-8")
(output / "config.json").write_text(json.dumps(overrides), encoding="utf-8")  # «модель»
