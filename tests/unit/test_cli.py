"""Тесты urec.cli без интернета: вместо TOFU — пары из фикстуры tests/fixtures/tofu_mini.jsonl."""

import json
from pathlib import Path

import pytest

from urec import cli
from urec.data.tofu import make_items
from urec.io import read_jsonl
from urec.types import QAItem

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tofu_mini.jsonl"


def fixture_items(name):
    """20 пар фикстуры как сплит name."""
    with open(FIXTURE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return make_items(name, rows, rows if name.startswith("forget") else None)


def test_prepare_data_writes_all_files(tmp_path, monkeypatch):
    calls = []  # с какими n и seed команда попросила retain_eval

    def fake_retain_eval(n, seed):
        calls.append((n, seed))
        return fixture_items("retain95")[:5]

    # подменяем загрузку TOFU: команда получит пары фикстуры, интернет не нужен
    monkeypatch.setattr(cli, "load_split", fixture_items)
    monkeypatch.setattr(cli, "build_retain_eval", fake_retain_eval)
    monkeypatch.setattr(cli, "build_retain_pool", lambda chosen: fixture_items("retain95")[5:])

    cli.main(["prepare-data", f"paths.data={tmp_path.as_posix()}", "data.retain_eval_seed=3"])

    assert calls == [(400, 3)]  # размер — из конфига, сид — из командной строки

    items_dir = tmp_path / "items"
    names = ["forget01", "forget05", "forget10", "retain_eval", "retain_pool"]
    assert sorted(path.name for path in items_dir.iterdir()) == [f"{n}.jsonl" for n in names]
    assert read_jsonl(items_dir / "forget05.jsonl", QAItem) == fixture_items("forget05")
    assert len(read_jsonl(items_dir / "retain_eval.jsonl", QAItem)) == 5
    assert len(read_jsonl(items_dir / "retain_pool.jsonl", QAItem)) == 15


@pytest.mark.parametrize("args", [[], ["prepare-dat"]], ids=["no-command", "typo"])
def test_unknown_command_prints_usage(args):
    with pytest.raises(SystemExit, match="Команды: prepare-data"):
        cli.main(args)
