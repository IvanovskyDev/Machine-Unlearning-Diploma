"""Клиенты серверов vLLM (план, блок 26): запросы к victim и к атакующему.

Сервер vLLM говорит на том же протоколе, что API OpenAI, поэтому запросы идут через
библиотеку openai. Victim получает промпты готовыми token ids (/v1/completions),
атакующий и судья — обычные диалоги (/v1/chat/completions).
"""

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError
from openai.types.chat import ChatCompletionMessageParam
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

# повторять запрос при сбоях сети, таймауте и ошибке сервера (ответы 5xx)
RETRY_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError)

Message = ChatCompletionMessageParam  # одно сообщение диалога: {"role": …, "content": …}


@dataclass
class GenParams:
    """Параметры генерации: как отвечать и когда остановиться."""

    temperature: float  # 0 — всегда самый вероятный токен (жадная генерация)
    top_p: float  # выбирать только из самых вероятных токенов с суммарной вероятностью top_p
    max_tokens: int  # ответ не длиннее
    seed: int  # сид случайной выборки на сервере
    stop_token_ids: list[int] | None = None  # на каких токенах остановиться


@dataclass
class PerfStats:
    """Счётчики клиента: сколько запросов, повторов, токенов и времени — для perf.json."""

    requests: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: list[float] = field(default_factory=list)  # длительность каждого запроса

    def summary(self) -> dict[str, float]:
        """Итог счётчиков одним словарём."""
        ordered = sorted(self.seconds) or [0.0]
        return {
            "requests": self.requests,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds_total": sum(ordered),
            "seconds_median": ordered[len(ordered) // 2],
            "seconds_max": ordered[-1],
        }


def load_stop_token_ids(model_dir: str | Path) -> list[int]:
    """Токены конца ответа из generation_config.json модели — кодом, а не руками (блок 26)."""
    config = json.loads((Path(model_dir) / "generation_config.json").read_text(encoding="utf-8"))
    eos = config["eos_token_id"]  # число или список чисел
    return list(eos) if isinstance(eos, list) else [eos]


def request_seed(*parts: object) -> int:
    """Сид запроса из его «адреса», например (run, item_id, node_id, попытка).

    Один адрес — всегда один сид, в каком бы порядке сервер ни отвечал, поэтому выборка
    атакующего воспроизводима. hash() Python не годится: он меняется от запуска к запуску.
    """
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 2**31  # неотрицательное число до 2**31


class _Client:
    """Общее у клиентов: подключение к серверу, повторы при сбоях и счётчики."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        max_attempts: int = 5,
        retry_wait: float = 1.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.model = model  # имя модели на сервере (--served-model-name)
        # max_retries=0: повторами управляет tenacity ниже, у всех запросов одинаково
        self.api = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",  # серверу vLLM ключ не нужен, но библиотека требует строку
            timeout=timeout,
            max_retries=0,
            http_client=http_client,  # в тестах — поддельный сервер
        )
        self.max_attempts = max_attempts
        self.retry_wait = retry_wait
        self.perf = PerfStats()

    async def _call(self, send: Callable[[], Awaitable[Any]]) -> Any:
        """Выполняет запрос send(); при сбое повторяет с растущей паузой, до max_attempts раз."""
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(RETRY_ERRORS),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=self.retry_wait, max=60),  # 1, 2, 4, … секунд
            reraise=True,  # попытки кончились — выдать исходную ошибку
        )
        async for attempt in retrying:
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    self.perf.retries += 1
                start = time.perf_counter()
                response = await send()
        self.perf.requests += 1
        self.perf.seconds.append(time.perf_counter() - start)
        if response.usage is not None:  # сколько токенов прочитал и написал сервер
            self.perf.prompt_tokens += response.usage.prompt_tokens
            self.perf.completion_tokens += response.usage.completion_tokens
        return response


class VictimClient(_Client):
    """Victim: промпты готовыми token ids через /v1/completions (план, блок 26)."""

    def __init__(
        self,
        base_url: str,
        model: str = "victim",
        batch_size: int = 64,
        parallel: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, model, **kwargs)
        self.batch_size = batch_size  # промптов в одном запросе
        self.parallel = parallel  # запросов одновременно

    async def generate(self, prompt_ids: list[list[int]], params: GenParams) -> list[str]:
        """Ответы victim на промпты prompt_ids — в том же порядке, что промпты."""
        semaphore = asyncio.Semaphore(self.parallel)  # не больше parallel запросов сразу

        async def one_batch(batch: list[list[int]]) -> list[str]:
            async with semaphore:
                response = await self._call(
                    lambda: self.api.completions.create(
                        model=self.model,
                        prompt=batch,  # список промптов, каждый — список token ids
                        max_tokens=params.max_tokens,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        seed=params.seed,
                        extra_body=_stop_body(params),
                    )
                )
            if len(response.choices) != len(batch):
                raise RuntimeError(f"промптов {len(batch)}, а ответов {len(response.choices)}")
            # сервер может вернуть ответы не по порядку: index — номер промпта в запросе
            choices = sorted(response.choices, key=lambda choice: choice.index)
            return [choice.text for choice in choices]

        size = self.batch_size
        batches = [prompt_ids[start : start + size] for start in range(0, len(prompt_ids), size)]
        # gather запускает пачки вместе и возвращает ответы в порядке пачек, а не готовности
        answers = await asyncio.gather(*(one_batch(batch) for batch in batches))
        return [text for batch_answers in answers for text in batch_answers]


class ChatClient(_Client):
    """Атакующий и судья: диалоги через /v1/chat/completions, один диалог на запрос."""

    def __init__(self, base_url: str, model: str, parallel: int = 64, **kwargs: Any) -> None:
        super().__init__(base_url, model, **kwargs)
        self.parallel = parallel  # запросов одновременно

    async def chat(
        self, convs: list[list[Message]], params: GenParams, seeds: list[int] | None = None
    ) -> list[str]:
        """Ответы на диалоги convs — в том же порядке. seeds — свой сид каждому диалогу."""
        if seeds is None:
            seeds = [params.seed] * len(convs)
        if len(seeds) != len(convs):
            raise ValueError(f"диалогов {len(convs)}, а сидов {len(seeds)}")
        semaphore = asyncio.Semaphore(self.parallel)

        async def one(conv: list[Message], seed: int) -> str:
            async with semaphore:
                response = await self._call(
                    lambda: self.api.chat.completions.create(
                        model=self.model,
                        messages=conv,
                        max_tokens=params.max_tokens,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        seed=seed,
                        extra_body=_stop_body(params),
                    )
                )
            return response.choices[0].message.content or ""  # пустой ответ — пустая строка

        return list(await asyncio.gather(*(one(conv, seed) for conv, seed in zip(convs, seeds))))


def _stop_body(params: GenParams) -> dict[str, Any]:
    """Параметры vLLM сверх API OpenAI: stop_token_ids (токены конца ответа)."""
    return {"stop_token_ids": params.stop_token_ids} if params.stop_token_ids else {}
