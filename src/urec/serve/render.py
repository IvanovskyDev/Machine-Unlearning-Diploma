"""Промпт victim (план, блок 26): те же token ids, что видит модель в оценке OpenUnlearning.

Victim получает промпт готовыми token ids, а не строкой: строка уже начинается с
<|begin_of_text|>, и сервер vLLM добавил бы второй такой токен — это незаметно меняет
ответы. Ids строятся той же логикой, что preprocess_chat_instance(...,
predict_with_generate=True) в форке, а золотой тест сверяет их с выгрузкой форка
tests/golden/prompt_ids.json.
"""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

TemplateArgs = dict[str, Any]  # настройки шаблона диалога из конфига модели форка


def load_template_args(fork_model_yaml: str | Path) -> TemplateArgs:
    """template_args из конфига модели форка, например configs/model/Llama-3.2-1B-Instruct.yaml.

    Один источник правды: шаблон берётся оттуда же, откуда его берёт оценка форка.
    """
    template_args = OmegaConf.to_container(OmegaConf.load(fork_model_yaml).template_args)
    if not isinstance(template_args, dict):
        raise ValueError(f"{fork_model_yaml}: template_args — не словарь")
    return {str(key): value for key, value in template_args.items()}


def render_victim_ids(
    tokenizer: Any, template_args: TemplateArgs, prompts: list[str], responses: list[str]
) -> list[int]:
    """Token ids промпта victim: system, завершённые ходы и последний вопрос.

    prompts — вопросы q_1 … q_t, responses — ответы y_1 … y_(t-1) на все вопросы,
    кроме последнего: на него victim ответит сейчас.
    """
    if len(responses) != len(prompts) - 1:
        raise ValueError(f"вопросов {len(prompts)}, ответов {len(responses)}: нужно на один меньше")
    if not template_args["apply_chat_template"]:
        raise NotImplementedError("поддерживаются модели с apply_chat_template: True, как Llama")

    chat = []  # диалог в формате шаблона: список {"role": …, "content": …}
    if template_args.get("system_prompt"):
        chat.append({"role": "system", "content": template_args["system_prompt"]})
    for prompt, response in zip(prompts, responses):  # zip кончается на последнем ответе
        chat.append({"role": "user", "content": prompt})
        chat.append({"role": "assistant", "content": response})
    chat.append({"role": "user", "content": prompts[-1]})  # вопрос, на который victim ответит

    date = template_args.get("date_string")  # дата в шаблоне Llama; без неё — сегодняшняя
    date_info = {"date_string": date} if date is not None else {}
    encoded = tokenizer.apply_chat_template(
        chat,
        tokenize=True,  # сразу token ids, а не строка
        add_generation_prompt=True,  # в конце — заголовок ответа ассистента
        return_dict=True,  # словарь с "input_ids" — одинаково во всех версиях transformers
        **date_info,
    )
    return list(encoded["input_ids"])


def truncate_turns(
    tokenizer: Any,
    template_args: TemplateArgs,
    prompts: list[str],
    responses: list[str],
    max_prompt_tokens: int,
) -> tuple[list[str], list[str], bool]:
    """Выбрасывает самые ранние целые ходы, пока промпт длиннее max_prompt_tokens.

    System и последний вопрос остаются всегда. Возвращает (prompts, responses, truncated),
    где truncated — выброшен ли хотя бы один ход (флаг truncated_context в Node).
    """
    truncated = False
    while len(prompts) > 1:
        ids = render_victim_ids(tokenizer, template_args, prompts, responses)
        if len(ids) <= max_prompt_tokens:
            break
        prompts, responses = prompts[1:], responses[1:]  # самый ранний ход: вопрос и ответ
        truncated = True
    return prompts, responses, truncated
