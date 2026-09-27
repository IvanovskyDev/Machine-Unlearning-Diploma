"""Данные TOFU (план, блоки 13 и 25): сплиты как списки QAItem, имена авторов, наборы retain.

TOFU — 200 вымышленных авторов по 20 пар «вопрос — ответ»; пары одного автора идут подряд.
Каждый сплит — один файл JSON Lines в датасете locuslab/TOFU на Hugging Face. Файлы
скачиваются через huggingface_hub по закреплённой версии TOFU_REVISION: если авторы датасета
что-то в нём поменяют, у нас данные останутся прежними.
"""

import json
import random
from collections import Counter
from typing import Any

from huggingface_hub import hf_hub_download

from urec.types import QAItem

TOFU_REPO = "locuslab/TOFU"  # датасет на Hugging Face
TOFU_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"  # его версия (коммит) от 27.03.2025
PAIRS_PER_AUTHOR = 20  # у каждого автора 20 пар подряд (блок 13)
RETAIN_SPLIT = "retain95"  # из него строятся retain_eval и retain_pool (блоки 25 и 30)

# настройки extract_author
NAME_PARTICLES = {"van", "der", "de", "von"}  # строчные частицы внутри имён: Isabella van Pletzen
MIN_NAME_WORDS = 2  # имя — от 2 до 4 слов
MAX_NAME_WORDS = 4
MIN_NAME_SHARE = 0.5  # имя должно встретиться хотя бы в половине пар автора

Row = dict[str, Any]  # одна строка файла TOFU: {"question": …, "answer": …, …}


def read_tofu(name: str) -> list[Row]:
    """Скачивает файл сплита TOFU (один раз, дальше — из кэша) и возвращает его строки."""
    path = hf_hub_download(TOFU_REPO, f"{name}.json", repo_type="dataset", revision=TOFU_REVISION)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]  # пустые строки пропускаются


def load_split(name: str) -> list[QAItem]:
    """Сплит TOFU как список QAItem; у forget-сплитов — с ответами из {name}_perturbed."""
    rows = read_tofu(name)
    perturbed = read_tofu(f"{name}_perturbed") if name.startswith("forget") else None
    return make_items(name, rows, perturbed)


def make_items(name: str, rows: list[Row], perturbed: list[Row] | None = None) -> list[QAItem]:
    """Строки сплита name → список QAItem. perturbed — строки {name}_perturbed или None."""
    if len(rows) % PAIRS_PER_AUTHOR != 0:
        raise ValueError(f"{name}: {len(rows)} строк не делится на {PAIRS_PER_AUTHOR}")
    extra: list[Row | None] = [None] * len(rows)  # строка из _perturbed для каждой строки
    if perturbed is not None:
        extra = list(match_perturbed(rows, perturbed))

    items = []
    for index, (row, more) in enumerate(zip(rows, extra)):
        item = QAItem(
            item_id=f"{name}/{index:04d}",  # "forget05/0137": сплит и номер строки
            split=name,
            index=index,
            author_local=index // PAIRS_PER_AUTHOR,  # // — деление нацело
            author=None,  # имя ставится ниже, одно на 20 пар автора
            question=row["question"],
            answer=row["answer"],
            paraphrased_answer=more["paraphrased_answer"] if more else None,
            perturbed_answers=tuple(more["perturbed_answer"]) if more else (),
            keys=(),  # ключевые факты появятся в вехе M2
            evaluable=True,  # в M0 временно True; настоящая проверка — в вехе M2
        )
        items.append(item)

    for start in range(0, len(items), PAIRS_PER_AUTHOR):
        block = items[start : start + PAIRS_PER_AUTHOR]  # 20 пар одного автора
        author = extract_author(block)
        for item in block:
            item.author = author
    return items


def match_perturbed(rows: list[Row], perturbed: list[Row]) -> list[Row]:
    """Для каждой строки rows находит строку perturbed с тем же вопросом.

    Порядок строк в двух файлах не обязан совпадать (блок 25), поэтому пары ищутся
    по тексту вопроса. Любое несовпадение — ошибка: сопоставление должно быть без потерь.
    """
    by_question = {row["question"]: row for row in perturbed}
    if len(by_question) != len(perturbed):
        raise ValueError("в _perturbed повторяются вопросы: сопоставить по вопросу нельзя")
    if len({row["question"] for row in rows}) != len(rows):
        raise ValueError("в сплите повторяются вопросы: сопоставить по вопросу нельзя")
    if len(rows) != len(perturbed):
        raise ValueError(f"в сплите {len(rows)} строк, а в _perturbed {len(perturbed)}")

    matched = []
    for row in rows:
        other = by_question.get(row["question"])
        if other is None:
            raise ValueError(f"в _perturbed нет вопроса {row['question']!r}")
        if other["answer"] != row["answer"]:
            raise ValueError(f"у вопроса {row['question']!r} в _perturbed другой ответ")
        matched.append(other)
    return matched


def extract_author(items: list[QAItem]) -> str | None:
    """Имя автора по его парам «вопрос — ответ»; None, если имени в них нет.

    Кандидаты — куски из 2–4 подряд идущих слов с заглавной буквы (внутри могут быть
    частицы вроде «van» и прозвище в скобках). Побеждает кусок, который встречается в
    большем числе пар; при равенстве — более длинный: полное имя, а не его часть.
    """
    counts: Counter[str] = Counter()
    for item in items:
        # множество: кусок, который встретился в паре дважды, считается один раз
        counts.update(_name_pieces(item.question) | _name_pieces(item.answer))
    if not counts:
        return None
    name, pairs = max(counts.items(), key=lambda kv: (kv[1], len(kv[0].split())))
    if pairs < MIN_NAME_SHARE * len(items):
        return None  # например, автора везде называют только «the author»
    return name


def retain_candidates() -> list[QAItem]:
    """Пары retain95 без пар retain_perturbed.

    Пары retain_perturbed использует оценка TOFU в OpenUnlearning (метрики retain в
    Model Utility), поэтому в наши наборы они не входят. Пары сравниваются по вопросу
    вместе с ответом: вопрос «What is the full name of the author?» есть у двух авторов.
    """
    excluded = {(row["question"], row["answer"]) for row in read_tofu("retain_perturbed")}
    retain = load_split(RETAIN_SPLIT)
    return [item for item in retain if (item.question, item.answer) not in excluded]


def build_retain_eval(n: int = 400, seed: int = 0) -> list[QAItem]:
    """D_r^eval: n случайных пар из retain_candidates(); тот же seed — тот же набор."""
    chosen = random.Random(seed).sample(retain_candidates(), n)  # свой генератор с сидом seed
    return sorted(chosen, key=lambda item: item.index)  # по порядку строк в retain95


def build_retain_pool(retain_eval: list[QAItem]) -> list[QAItem]:
    """Пул для retain-режимов (блок 30): retain_candidates() без пар retain_eval."""
    taken = {item.item_id for item in retain_eval}
    return [item for item in retain_candidates() if item.item_id not in taken]


def _name_pieces(text: str) -> set[str]:
    """Все куски из 2–4 подряд идущих слов имени в тексте."""
    pieces = set()
    for run in _name_runs(text):
        for size in range(MIN_NAME_WORDS, MAX_NAME_WORDS + 1):
            for start in range(len(run) - size + 1):
                piece = run[start : start + size]
                # кусок начинается и кончается словом с заглавной буквы, а не частицей или прозвищем
                if _is_capitalized(piece[0]) and _is_capitalized(piece[-1]):
                    pieces.add(" ".join(piece))
    return pieces


def _name_runs(text: str) -> list[list[str]]:
    """Цепочки подряд идущих слов, похожих на части имени."""
    runs = []
    run: list[str] = []
    for token in text.split():
        word = token.rstrip(".,;:!?\"”'")  # знаки препинания и кавычки в конце слова
        ends_here = word != token  # после знака препинания имя не продолжается
        if word.endswith(("'s", "’s")):  # Al-Kuwaiti's → Al-Kuwaiti
            word = word[:-2]
            ends_here = True
        in_name = word in NAME_PARTICLES or _is_nickname(word)
        if _is_capitalized(word) or (run and in_name):
            run.append(word)
            if ends_here:
                runs.append(run)
                run = []
        else:
            runs.append(run)
            run = []
    runs.append(run)
    return [run for run in runs if len(run) >= MIN_NAME_WORDS]


def _is_capitalized(word: str) -> bool:
    """Слово начинается с заглавной буквы: Basil, Al-Kuwaiti, Beltrán."""
    return word[:1].isupper()


def _is_nickname(word: str) -> bool:
    """Прозвище в скобках внутри имени: Alejandro (Alex) Fuentes."""
    return word.startswith("(") and word.endswith(")") and word[1:2].isupper()
