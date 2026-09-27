"""Тесты urec.types: правила, которые классы проверяют сами."""

from dataclasses import replace

import pytest

from urec.types import StepStatus

from sample_records import ALL_RECORDS, HARD_EXAMPLE, MANIFEST, QA_ITEM  # образцы записей


def test_every_class_has_schema_version_1():
    # правило блока 22: у каждого класса есть schema_version, сейчас у всех версия 1
    for record in ALL_RECORDS:
        assert record.schema_version == 1


def test_qa_item_turns_lists_into_tuples():
    # dataclasses.replace — копия объекта с другими значениями указанных полей
    item = replace(QA_ITEM, perturbed_answers=["a", "b"], keys=["Kuwait City"])
    assert item.perturbed_answers == ("a", "b")
    assert item.keys == ("Kuwait City",)


def test_hard_example_needs_one_response_less_than_prompts():
    with pytest.raises(ValueError, match="на один меньше"):
        replace(HARD_EXAMPLE, responses=["ответ 1", "ответ 2"])  # вопросов 2, ответов тоже 2


def test_manifest_turns_dicts_into_step_status():
    step = {
        "started": "2026-10-14T12:30:05+03:00",
        "finished": None,
        "status": "running",
        "seconds": None,
        "command": ["python", "src/eval.py"],
        "peak_gpu_mib": None,
    }
    manifest = replace(MANIFEST, steps={"eval": step})
    assert manifest.steps["eval"] == StepStatus(
        "2026-10-14T12:30:05+03:00", None, "running", None, ["python", "src/eval.py"], None
    )
