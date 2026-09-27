"""prepare-data на настоящем TOFU даёт ровно те файлы data/items, что лежат в репозитории.

Нужен интернет. Если тест упал, значит, код или данные изменились и файлы в репозитории
устарели: либо это ошибка, либо файлы нужно записать заново и объяснить это в PR.
"""

import pytest

from urec import cli
from urec.config import REPO_ROOT

pytestmark = pytest.mark.network

NAMES = ["forget01", "forget05", "forget10", "retain_eval", "retain_pool"]


def test_prepare_data_reproduces_committed_files(tmp_path):
    cli.main(["prepare-data", f"paths.data={tmp_path.as_posix()}"])  # пишет в tmp_path/items
    for name in NAMES:
        new = (tmp_path / "items" / f"{name}.jsonl").read_bytes()
        committed = (REPO_ROOT / "data" / "items" / f"{name}.jsonl").read_bytes()
        assert new == committed, f"{name}.jsonl отличается от файла в репозитории"
