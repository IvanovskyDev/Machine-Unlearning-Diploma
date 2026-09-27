"""Проверка файлов data/items в репозитории (план, блок 25) — без интернета.

Совпадают ли файлы с тем, что пишет prepare-data, проверяет
tests/integration/test_prepare_data.py; здесь — что файлы целы и согласованы.
"""

import pytest

from urec.config import REPO_ROOT
from urec.io import read_jsonl
from urec.types import QAItem

ITEMS = REPO_ROOT / "data" / "items"
SIZES = {"forget01": 40, "forget05": 200, "forget10": 400, "retain_eval": 400, "retain_pool": 3000}


@pytest.mark.parametrize("name, size", SIZES.items())
def test_file_is_readable_and_has_expected_size(name, size):
    items = read_jsonl(ITEMS / f"{name}.jsonl", QAItem)  # read_jsonl проверяет версию и поля
    assert len(items) == size
    assert len({item.item_id for item in items}) == size  # item_id не повторяются


def test_retain_eval_and_pool_do_not_overlap():
    retain_eval = {item.item_id for item in read_jsonl(ITEMS / "retain_eval.jsonl", QAItem)}
    pool = {item.item_id for item in read_jsonl(ITEMS / "retain_pool.jsonl", QAItem)}
    assert not retain_eval & pool


def test_forget_splits_are_nested():
    # в TOFU forget01 — последние 2 автора, forget05 — последние 10, forget10 — последние 20
    questions = {}
    for name in ["forget01", "forget05", "forget10"]:
        questions[name] = {item.question for item in read_jsonl(ITEMS / f"{name}.jsonl", QAItem)}
    assert questions["forget01"] < questions["forget05"] < questions["forget10"]
