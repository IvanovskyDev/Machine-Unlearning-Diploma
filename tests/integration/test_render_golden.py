"""Золотой тест промпта victim (план, блок 26): render_victim_ids даёт те же token ids, что форк.

Эталон tests/golden/prompt_ids.json записал scripts/export_prompts.py форка в окружении
с версиями unl: 40 вопросов forget01 и 20 многоходовых диалогов. Нужен интернет —
токенизатор модели скачивается с Hugging Face по ревизии из envs/models.lock.json.
"""

import json

import pytest
from huggingface_hub import HfApi

from urec.config import REPO_ROOT
from urec.serve.render import load_template_args, render_victim_ids

pytestmark = pytest.mark.network

GOLDEN = json.loads((REPO_ROOT / "tests/golden/prompt_ids.json").read_text(encoding="utf-8"))
LOCK = json.loads((REPO_ROOT / "envs/models.lock.json").read_text(encoding="utf-8"))
FORK_MODEL_YAML = REPO_ROOT / "external/open-unlearning/configs/model/Llama-3.2-1B-Instruct.yaml"
MODEL_1B = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
MODEL_3B = "open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer  # тяжёлая библиотека: импорт внутри

    return AutoTokenizer.from_pretrained(MODEL_1B, revision=LOCK[MODEL_1B])


def test_golden_was_written_for_this_tokenizer_and_template():
    assert GOLDEN["meta"]["tokenizer"] == MODEL_1B
    assert GOLDEN["meta"]["tokenizer_revision"] == LOCK[MODEL_1B]
    assert GOLDEN["meta"]["template_args"] == load_template_args(FORK_MODEL_YAML)
    kinds = [example["kind"] for example in GOLDEN["examples"]]
    assert kinds.count("forget01") == 40 and kinds.count("dialog") == 20


def test_render_matches_fork_token_by_token(tokenizer):
    template_args = load_template_args(FORK_MODEL_YAML)
    for number, example in enumerate(GOLDEN["examples"]):
        ids = render_victim_ids(tokenizer, template_args, example["prompts"], example["responses"])
        assert ids == example["ids"], f"пример {number} ({example['kind']})"
        assert ids.count(tokenizer.bos_token_id) == 1  # ровно один <|begin_of_text|>


def test_3b_tokenizer_is_the_same_as_1b():
    # у одинаковых файлов в git Hugging Face одинаковый blob_id: тогда эталон годится и для 3B
    def blob_ids(model):
        info = HfApi().model_info(model, revision=LOCK[model], files_metadata=True)
        return {s.rfilename: s.blob_id for s in info.siblings if s.rfilename in TOKENIZER_FILES}

    assert blob_ids(MODEL_3B) == blob_ids(MODEL_1B)
