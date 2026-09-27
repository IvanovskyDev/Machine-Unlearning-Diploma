"""Команды urec (план, блоки 21 и 25): python -m urec.cli <команда> [ключ=значение ...].

После имени команды можно поменять любое значение конфига (configs/), например:
    python -m urec.cli prepare-data paths.data=/content/check
"""

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from omegaconf import DictConfig

from urec.config import load_config
from urec.data.tofu import build_retain_eval, build_retain_pool, load_split
from urec.io import write_jsonl
from urec.types import QAItem


def prepare_data(cfg: DictConfig) -> None:
    """Пишет вопросы TOFU в {paths.data}/items: forget-сплиты, retain_eval и retain_pool."""
    files: dict[str, list[QAItem]] = {}  # имя файла без .jsonl → его записи
    for split in cfg.data.forget_splits:
        files[split] = load_split(split)
    files["retain_eval"] = build_retain_eval(cfg.data.retain_eval_size, cfg.data.retain_eval_seed)
    files["retain_pool"] = build_retain_pool(files["retain_eval"])

    items_dir = Path(cfg.paths.data) / "items"
    for name, items in files.items():
        path = items_dir / f"{name}.jsonl"
        write_jsonl(path, items)
        print(f"{name}: {len(items)} вопросов → {path}")


COMMANDS: dict[str, Callable[[DictConfig], None]] = {  # имя команды → функция
    "prepare-data": prepare_data,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Разбирает командную строку, собирает конфиг и запускает команду."""
    args = list(sys.argv[1:] if argv is None else argv)  # слова после «python -m urec.cli»
    if not args or args[0] not in COMMANDS:
        raise SystemExit(
            "Запуск: python -m urec.cli <команда> [ключ=значение ...]\n"
            f"Команды: {', '.join(COMMANDS)}"
        )
    command, *overrides = args
    COMMANDS[command](load_config(overrides))


if __name__ == "__main__":  # файл запущен как программа: python -m urec.cli …
    main()
