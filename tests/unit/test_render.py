"""Тесты urec.serve.render без интернета: вместо настоящего токенизатора — поддельный.

Совпадение с токенами форка проверяет tests/integration/test_render_golden.py.
"""

import pytest

from urec.config import REPO_ROOT
from urec.serve.render import load_template_args, render_victim_ids, truncate_turns

FORK_MODEL_YAML = REPO_ROOT / "external/open-unlearning/configs/model/Llama-3.2-1B-Instruct.yaml"
TEMPLATE_ARGS = {"apply_chat_template": True, "system_prompt": "sys", "date_string": "10 Apr 2025"}


class FakeTokenizer:
    """Поддельный токенизатор: запоминает, что ему передали, и даёт по «токену» на слово."""

    def __init__(self):
        self.calls = []  # (диалог, остальные аргументы) каждого вызова

    def apply_chat_template(self, chat, **kwargs):
        self.calls.append((chat, kwargs))
        words = sum(len(message["content"].split()) for message in chat)
        return {"input_ids": list(range(words))}


def test_load_template_args_reads_fork_config():
    template_args = load_template_args(FORK_MODEL_YAML)
    assert template_args["apply_chat_template"] is True
    assert template_args["system_prompt"] == "You are a helpful assistant."
    assert template_args["date_string"] == "10 Apr 2025"


def test_render_builds_the_chat_like_the_fork():
    tokenizer = FakeTokenizer()
    render_victim_ids(tokenizer, TEMPLATE_ARGS, ["q1", "q2", "q3"], ["a1", "a2"])
    chat, kwargs = tokenizer.calls[0]
    assert chat == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},  # последний вопрос — без ответа
    ]
    assert kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "date_string": "10 Apr 2025",
    }


def test_render_without_system_prompt_and_date():
    tokenizer = FakeTokenizer()
    render_victim_ids(tokenizer, {"apply_chat_template": True}, ["q1"], [])
    chat, kwargs = tokenizer.calls[0]
    assert chat == [{"role": "user", "content": "q1"}]
    assert "date_string" not in kwargs


def test_render_needs_one_response_less_than_prompts():
    with pytest.raises(ValueError, match="на один меньше"):
        render_victim_ids(FakeTokenizer(), TEMPLATE_ARGS, ["q1", "q2"], ["a1", "a2"])


def test_render_needs_chat_template():
    with pytest.raises(NotImplementedError):
        render_victim_ids(FakeTokenizer(), {"apply_chat_template": False}, ["q1"], [])


# «токены» поддельного токенизатора — слова: sys (1) + ход 1 (4 + 3) + ход 2 (2 + 1) + q3 (1) = 12
PROMPTS = ["q1 w w w", "q2 w", "q3"]
RESPONSES = ["a1 w w", "a2"]


def test_truncate_keeps_everything_that_fits():
    assert truncate_turns(FakeTokenizer(), TEMPLATE_ARGS, PROMPTS, RESPONSES, 12) == (
        PROMPTS,
        RESPONSES,
        False,
    )


def test_truncate_drops_the_earliest_turn():
    # без хода 1 остаётся 1 + (2 + 1) + 1 = 5 «токенов» — влезает в 8
    assert truncate_turns(FakeTokenizer(), TEMPLATE_ARGS, PROMPTS, RESPONSES, 8) == (
        ["q2 w", "q3"],
        ["a2"],
        True,
    )


def test_truncate_always_keeps_the_last_question():
    # даже последний вопрос с system не влезает в 1 «токен», но вопрос остаётся
    assert truncate_turns(FakeTokenizer(), TEMPLATE_ARGS, PROMPTS, RESPONSES, 1) == (
        ["q3"],
        [],
        True,
    )
