"""Тесты urec.data.tofu без интернета.

Фикстура tests/fixtures/tofu_mini.jsonl — первые 20 строк forget01_perturbed из TOFU
(версия TOFU_REVISION, лицензия MIT): все пары первого автора forget01 с перефразированными
и неверными ответами. Остальные тексты в тестах тоже взяты из TOFU.
"""

import json
import random
from pathlib import Path

import pytest

from urec.data.tofu import extract_author, make_items

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tofu_mini.jsonl"
AUTHOR = "Basil Mahfouz Al-Kuwaiti"  # автор всех 20 пар фикстуры


def read_fixture():
    """Строки фикстуры — с теми же полями, что в forget01_perturbed."""
    with open(FIXTURE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def plain(rows):
    """Те же строки, как в обычном сплите forget01: только вопрос и ответ."""
    return [{"question": row["question"], "answer": row["answer"]} for row in rows]


def test_make_items_fills_all_fields():
    rows = read_fixture()
    items = make_items("forget01", plain(rows), rows)
    assert [item.item_id for item in items] == [f"forget01/{i:04d}" for i in range(20)]
    assert [item.index for item in items] == list(range(20))
    assert {item.split for item in items} == {"forget01"}
    assert {item.author_local for item in items} == {0}  # все 20 пар — один автор
    assert {item.author for item in items} == {AUTHOR}
    assert all(item.paraphrased_answer and len(item.perturbed_answers) == 5 for item in items)
    assert all(item.keys == () and item.evaluable for item in items)  # так до вехи M2


def test_perturbed_rows_are_matched_by_question_not_by_order():
    rows = read_fixture()
    shuffled = rows.copy()
    random.Random(0).shuffle(shuffled)  # те же строки _perturbed, но в другом порядке
    expected = make_items("forget01", plain(rows), rows)
    assert make_items("forget01", plain(rows), shuffled) == expected


def test_missing_perturbed_row_is_an_error():
    rows = read_fixture()
    with pytest.raises(ValueError, match="в _perturbed 19"):
        make_items("forget01", plain(rows), rows[:-1])  # одной строки _perturbed не хватает


def test_different_answer_in_perturbed_is_an_error():
    rows = read_fixture()
    changed = [dict(row) for row in rows]  # копия строк, чтобы не портить исходные
    changed[3]["answer"] = "A different answer."
    with pytest.raises(ValueError, match="другой ответ"):
        make_items("forget01", plain(rows), changed)


def test_split_without_perturbed_has_no_extra_answers():
    items = make_items("forget01", plain(read_fixture()))
    assert all(item.paraphrased_answer is None and item.perturbed_answers == () for item in items)


def test_rows_must_come_in_blocks_of_20():
    with pytest.raises(ValueError, match="не делится на 20"):
        make_items("forget01", plain(read_fixture())[:19])


def test_extract_author_returns_the_full_name():
    items = make_items("forget01", plain(read_fixture()))
    assert extract_author(items) == AUTHOR  # все три слова, а не «Basil Mahfouz»


# kept — в скольких парах из 20 имя осталось; в остальных автора называют «the author»
@pytest.mark.parametrize("kept, expected", [(10, AUTHOR), (9, None)])
def test_name_must_be_in_at_least_half_of_the_pairs(kept, expected):
    rows = plain(read_fixture())
    for row in rows[kept:]:
        row["question"] = row["question"].replace(AUTHOR, "the author")
        row["answer"] = row["answer"].replace(AUTHOR, "the author")
    assert make_items("forget01", rows)[0].author == expected


# пары из TOFU с непростыми формами имени; каждая повторена 20 раз, как 20 пар одного автора
NAME_FORMS = [
    (
        "When and where was Isabella van Pletzen born?",
        "Isabella van Pletzen was born on 1st April 1975 in Cape Town, South Africa.",
        "Isabella van Pletzen",  # частица «van» внутри имени
    ),
    (
        "Where was Alejandro (Alex) Fuentes born?",
        "Alex Fuentes was born in Madrid, Spain.",
        "Alejandro (Alex) Fuentes",  # прозвище в скобках
    ),
    (
        "How does Xin Lee Williams' personal identification as LGBTQ+ influence their work?",
        "Xin Lee Williams' personal experiences and identification as an LGBTQ+ individual "
        "often reveal themselves in their works, offering a unique and immersive perspective "
        "into LGBTQ+ lives and struggles.",
        "Xin Lee Williams",  # притяжательная форма с апострофом в конце: Williams'
    ),
]


@pytest.mark.parametrize(
    "question, answer, expected", NAME_FORMS, ids=["van", "nickname", "williams"]
)
def test_extract_author_handles_name_forms(question, answer, expected):
    rows = [{"question": question, "answer": answer}] * 20
    assert make_items("retain95", rows)[0].author == expected
