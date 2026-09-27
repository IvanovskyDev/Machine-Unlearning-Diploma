"""Контракты данных (план, блок 22): форматы файлов, которыми обмениваются части проекта.

Каждый класс описывает одну строку файла (или один файл). Меняются классы сначала
в плане, потом здесь.

У каждого класса есть поле schema_version: оно записывается в файл вместе с данными,
а urec.io при чтении проверяет, что версия та же, что в коде. Так старый или чужой
файл даёт понятную ошибку, а не молча портит числа.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class Record(Protocol):
    """Любая запись из этого файла: у каждой есть поле schema_version."""

    schema_version: int


@dataclass
class QAItem:
    """Вопрос TOFU с эталонным ответом — строка data/items/{split}.jsonl."""

    item_id: str  # "forget05/0137": сплит и номер строки; одинаков во всех файлах проекта
    split: str  # сплит TOFU, например "forget05"
    index: int  # номер строки в сплите на Hugging Face, с нуля
    author_local: int  # index // 20: номер автора внутри сплита (у автора 20 вопросов подряд)
    author: str | None  # имя автора, извлекается один раз
    question: str  # вопрос x
    answer: str  # эталонный ответ y*
    paraphrased_answer: str | None  # перефразированный ответ из {split}_perturbed; None, если нет
    perturbed_answers: tuple[str, ...]  # неверные ответы из {split}_perturbed; пусто, если нет
    keys: tuple[str, ...]  # ключевые факты K(y*); заполняет scoring.keys в вехе M2
    evaluable: bool  # False: фактов нет или ответ вида «details are not provided»
    schema_version: int = 1

    def __post_init__(self) -> None:
        # JSON не отличает tuple от list: после чтения файла здесь списки.
        # Приводим их к tuple, чтобы прочитанный объект был равен записанному.
        self.perturbed_answers = tuple(self.perturbed_answers)
        self.keys = tuple(self.keys)


@dataclass
class Node:
    """Один запрос атаки и ответ на него — строка attack/…/nodes.jsonl.zst."""

    run: str  # имя запуска, например "exp2_3B_f05_NPO_base_s1"
    regime: str  # режим атаки: A, A+, B, C, D или D-blind
    attacker: str  # атакующая модель, например "qwen7b"
    attack_seed: int  # сид атаки
    item_id: str  # какой вопрос атакуют
    node_id: int  # номер узла
    parent_id: int | None  # узел-родитель — предыдущий ход той же ветки; None, если его нет
    turn: int  # номер хода, с 1
    strategy: str | None  # подсказка атакующему (context, decompose, …); None, если её нет
    beam_rank: int | None  # ранг в луче поиска; None вне лучевого поиска
    query: str  # вопрос атакующего к victim
    response: str  # ответ victim
    R: float  # оценка восстановления R
    R_hat: float | None  # оценка R̂ без знания эталона (режим D-blind); None, если не считалась
    em_new: float  # доля ключевых фактов, названных в ответе и не подсказанных в вопросах
    bs: float  # BERTScore ответа с эталоном
    refusal: bool  # ответ — отказ
    keys_hit: list[str]  # ключевые факты, найденные в ответе
    queries_used: int  # сколько запросов к victim по этому вопросу израсходовано
    success: bool  # R ≥ θ: знание восстановлено
    truncated_context: bool  # контекст victim пришлось обрезать, чтобы он поместился
    schema_version: int = 1


@dataclass
class ItemResult:
    """Итог атаки по одному вопросу — строка attack/…/items.csv."""

    run: str  # имя запуска
    regime: str  # режим атаки
    item_id: str  # вопрос
    evaluable: bool  # False: вопрос не входит в метрики атаки
    success: bool  # знание восстановлено хотя бы раз
    q_star: int | None  # запросов израсходовано к успеху; None без успеха
    t_star: int | None  # ход, на котором успех; None без успеха
    best_R: float  # лучшая оценка R за атаку
    queries_used: int  # запросов израсходовано всего
    schema_version: int = 1


@dataclass
class HardExample:
    """Трудный диалог для обучения TAU — строка tau/round{k}/hard.jsonl.

    Поля повторяют аргументы preprocess_chat_instance(prompt_msgs, response_msgs)
    из OpenUnlearning: форк получает готовый многоходовый диалог без преобразований.
    """

    item_id: str  # вопрос TOFU, из атаки на который взят диалог
    turn: int  # ход t
    prompts: list[str]  # вопросы q_1 … q_t
    responses: list[str]  # ответы y_1 … y_(t-1): на один меньше, чем вопросов
    answer: str  # y* — финальный ответ, на нём считается лосс
    weight: float  # вес примера w_(i,t)
    ref_nll: float | None  # заполняет precompute_ref в окружении unl; до этого None
    schema_version: int = 1

    def __post_init__(self) -> None:
        # правило из блока 22: ответов ровно на один меньше, чем вопросов
        if len(self.responses) != len(self.prompts) - 1:
            raise ValueError(
                f"HardExample {self.item_id}: вопросов {len(self.prompts)}, "
                f"ответов {len(self.responses)}, а должно быть на один меньше"
            )


@dataclass
class RunSpec:
    """Описание одного запуска — results/raw/{run}/spec.yaml."""

    run: str  # имя запуска: {exp}_{model}_{split}_{method}_{variant}_s{seed}
    exp: str  # эксперимент, например "exp2"
    model_size: str  # размер модели, например "3B"
    split: str  # forget-сплит, например "forget05"
    method: str  # метод забывания, например "NPO"
    variant: str  # вариант метода, например "base"
    seed: int  # сид забывания
    base_model: str  # путь или HF id target-модели
    unlearn_overrides: list[str]  # Hydra-добавки для форка
    regimes: list[str]  # режимы атаки
    attackers: list[str]  # атакующие модели
    attack_seeds: list[int]  # сиды атаки
    retain_regime: str | None  # режим retain; None — обычный retain
    keep_checkpoint: bool  # True — чекпоинт сохранить, False — удалить после атак
    schema_version: int = 1


@dataclass
class StepStatus:
    """Состояние одного шага запуска; хранится внутри Manifest и версии своей не имеет."""

    started: str  # время начала с часовым поясом, например "2026-10-14T12:30:05+03:00"
    finished: str | None  # время конца; None, пока шаг идёт
    status: str  # "running", "done" или "failed"
    seconds: float | None  # сколько секунд шёл шаг; None, пока шаг идёт
    command: list[str]  # команда шага целиком: по ней шаг можно повторить (блок 37)
    peak_gpu_mib: int | None  # пик занятой памяти GPU, МиБ; None — GPU нет или шаг ещё идёт


@dataclass
class Manifest:
    """Что и как было запущено — results/raw/{run}/manifest.json."""

    run: str  # имя запуска
    git: dict[str, str]  # коммиты urec и форка
    lock_sha: str  # sha256 lock-файлов окружений
    model_revisions: dict[str, str]  # ревизии моделей на Hugging Face
    hardware: str  # GPU, драйвер, CUDA
    steps: dict[str, StepStatus]  # имя шага → его состояние
    schema_version: int = 1

    def __post_init__(self) -> None:
        # после чтения JSON состояния шагов — обычные словари: превращаем их в StepStatus
        self.steps = {name: _as_step_status(value) for name, value in self.steps.items()}


def _as_step_status(value: StepStatus | dict[str, Any]) -> StepStatus:
    """Словарь из JSON превращает в StepStatus, готовый StepStatus возвращает как есть."""
    if isinstance(value, dict):
        return StepStatus(**value)
    return value
