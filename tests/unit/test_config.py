"""Тесты urec.config: сборка конфига из configs/ и seed_everything."""

import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from hydra.errors import ConfigCompositionException

from urec.config import REPO_ROOT, load_config, seed_everything


def test_default_config():
    cfg = load_config()
    assert Path(cfg.paths.root) == REPO_ROOT  # папку репозитория urec.config нашёл сам
    assert Path(cfg.paths.data) == REPO_ROOT / "data"
    fork = REPO_ROOT / "external" / "open-unlearning"
    assert Path(cfg.paths.fork_dir) == fork
    assert Path(cfg.paths.checkpoints) == fork / "saves" / "unlearn"
    assert Path(cfg.paths.raw) == REPO_ROOT / "results" / "raw"
    assert cfg.envs.unl == "/content/envs/unl/bin/python"  # python окружения unl в Colab
    assert cfg.allow_dirty is False  # по умолчанию — только из чистого репозитория
    assert list(cfg.data.forget_splits) == ["forget01", "forget05", "forget10"]
    assert cfg.data.retain_eval_size == 400
    assert cfg.data.retain_eval_seed == 0


def test_overrides_change_values(tmp_path):
    cfg = load_config([f"paths.data={tmp_path.as_posix()}", "data.retain_eval_seed=1"])
    assert Path(cfg.paths.data) == tmp_path
    assert cfg.data.retain_eval_seed == 1


def test_typo_in_override_is_an_error():
    with pytest.raises(ConfigCompositionException):
        load_config(["data.retain_eval_sed=1"])  # опечатка: такого ключа в конфиге нет


def test_models_path_comes_from_environment(monkeypatch):
    monkeypatch.setenv("MODELS", "/content/fast/models")  # как в Colab
    assert load_config().paths.models == "/content/fast/models"
    monkeypatch.delenv("MODELS")  # переменной нет — значение None
    assert load_config().paths.models is None


def test_seed_everything_repeats_random_numbers():
    seed_everything(7)
    first = (random.random(), np.random.rand())
    seed_everything(7)
    assert (random.random(), np.random.rand()) == first


def test_seed_everything_seeds_torch_if_installed(monkeypatch):
    calls = []
    fake_torch = types.SimpleNamespace(manual_seed=calls.append)  # поддельный torch: запоминает сид
    monkeypatch.setitem(sys.modules, "torch", fake_torch)  # import torch вернёт его
    seed_everything(7)
    assert calls == [7]
