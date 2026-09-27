"""Конфиги и сиды (план, блоки 23 и 25).

load_config собирает конфиг из YAML-файлов папки configs/ через Hydra: так все
параметры живут в конфигах, а не в коде. seed_everything задаёт один сид всем
генераторам случайных чисел, чтобы запуск повторялся.
"""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/urec/config.py → папка репозитория
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(overrides: Sequence[str] = (), config_name: str = "main") -> DictConfig:
    """Собирает конфиг из configs/{config_name}.yaml и добавок вида «ключ=значение»."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    if cfg.paths.root is None:  # папка репозитория не задана — берём ту, где лежит этот код
        cfg.paths.root = REPO_ROOT.as_posix()
    return cfg


def seed_everything(seed: int) -> None:
    """Задаёт сид seed генераторам Python, NumPy и PyTorch (если PyTorch установлен)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # тяжёлая библиотека: импорт внутри функции (план, блок 21)
    except ImportError:
        return  # PyTorch нет, например на GitHub в CI
    torch.manual_seed(seed)  # и на CPU, и на всех GPU
