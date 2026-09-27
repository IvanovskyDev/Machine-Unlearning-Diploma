"""Тесты urec.data.tofu на настоящих данных TOFU (план, блок 25).

Нужен интернет: файлы скачиваются с Hugging Face (несколько мегабайт, дальше — из кэша).
Без интернета эти тесты пропускает команда pytest -m "not gpu and not network".
"""

import pytest

from urec.data.tofu import build_retain_eval, build_retain_pool, load_split, read_tofu

pytestmark = pytest.mark.network  # метка network у всех тестов этого файла


@pytest.fixture(scope="module")
def splits():
    """Сплиты загружаются один раз на весь файл тестов."""
    return {name: load_split(name) for name in ["forget01", "forget05", "forget10", "retain95"]}


@pytest.fixture(scope="module")
def retain_eval():
    return build_retain_eval()


# размеры сплитов по блоку 13 плана
SIZES = [("forget01", 40), ("forget05", 200), ("forget10", 400), ("retain95", 3800)]


@pytest.mark.parametrize("name, size", SIZES)
def test_split_sizes(splits, name, size):
    assert len(splits[name]) == size


@pytest.mark.parametrize("name", ["forget01", "forget05", "forget10"])
def test_forget_splits_get_all_perturbed_answers(splits, name):
    for item in splits[name]:
        assert item.paraphrased_answer and len(item.perturbed_answers) == 5, item.item_id


# имена, проверенные глазами по текстам TOFU: (сплит, номер автора в сплите) → имя.
# retain95 — авторы 0–189 полного TOFU, forget10 — авторы 180–199, forget01 — 198 и 199.
CHECKED_AUTHORS = {
    ("forget01", 0): "Basil Mahfouz Al-Kuwaiti",  # имя из трёх слов
    ("forget01", 1): "Nikolai Abilov",
    ("forget10", 0): "Hsiao Yun-Hwa",
    ("forget10", 11): "Xin Lee Williams",  # чаще всего пишется как «Williams'»
    ("retain95", 0): "Jaime Vasquez",
    ("retain95", 5): "Aurelio Beltrán",  # вопрос «What is the full name of the author?»
    ("retain95", 22): "Isabel Martínez",  # … есть и у этого автора
    ("retain95", 65): "Isabella van Pletzen",  # частица «van»
    ("retain95", 72): "Alejandro (Alex) Fuentes",  # прозвище в скобках
    ("retain95", 88): None,  # имени в данных нет: только «the author» и «the fictitious author»
    ("retain95", 121): "Sara van Dyke",
    ("retain95", 153): "Femke Van der Veen",  # имя из четырёх слов
}


def test_author_names_checked_by_eye(splits):
    for (name, author_local), expected in CHECKED_AUTHORS.items():
        assert splits[name][author_local * 20].author == expected, (name, author_local)


def test_only_one_author_has_no_name(splits):
    # первые пары каждого автора; retain95 и forget10 вместе покрывают всех 200 авторов
    firsts = splits["retain95"][::20] + splits["forget10"][::20]
    assert [item.item_id for item in firsts if item.author is None] == ["retain95/1760"]


def test_every_name_is_in_at_least_half_of_its_pairs(splits):
    for name in ["retain95", "forget10"]:
        items = splits[name]
        for start in range(0, len(items), 20):
            block = items[start : start + 20]
            author = block[0].author
            if author is None:
                continue
            found = sum(author in item.question or author in item.answer for item in block)
            assert found >= 10, (block[0].item_id, author, found)


def test_retain_eval_is_always_the_same(retain_eval):
    assert build_retain_eval() == retain_eval  # тот же сид — тот же набор
    assert build_retain_eval(seed=1) != retain_eval  # другой сид — другой набор


def test_retain_eval_and_pool_are_clean(retain_eval):
    pool = build_retain_pool(retain_eval)
    assert len(retain_eval) == 400
    assert len(pool) == 3000  # 3800 пар retain95 − 400 из retain_perturbed − 400 в retain_eval
    perturbed = {(row["question"], row["answer"]) for row in read_tofu("retain_perturbed")}
    for item in retain_eval + pool:
        assert item.split == "retain95"
        assert (item.question, item.answer) not in perturbed, item.item_id
    assert not {item.item_id for item in retain_eval} & {item.item_id for item in pool}
    assert [item.index for item in retain_eval] == sorted(item.index for item in retain_eval)
