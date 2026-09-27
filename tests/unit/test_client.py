"""Тесты urec.serve.client без GPU: вместо сервера vLLM — поддельный сервер внутри теста.

httpx2.MockTransport отдаёт запросы клиента функции FakeServer.handle, а не в сеть.
asyncio.run запускает асинхронную функцию и ждёт её результата.
"""

import asyncio
import json
import random

import httpx2
import pytest
from openai import InternalServerError

from urec.serve.client import (
    ChatClient,
    GenParams,
    VictimClient,
    load_stop_token_ids,
    request_seed,
)

GREEDY = GenParams(temperature=0.0, top_p=1.0, max_tokens=200, seed=0, stop_token_ids=[1, 2, 3])
USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


class FakeServer:
    """Поддельный сервер vLLM: перемешивает порядок ответов и запоминает все запросы."""

    def __init__(self, failures=0):
        self.bodies = []  # тела всех запросов
        self.failures = failures  # сколько первых запросов ответить ошибкой 500
        self.in_flight = 0  # сколько запросов обрабатывается сейчас
        self.max_in_flight = 0  # и сколько было одновременно в худший момент

    async def handle(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.failures > 0:
            self.failures -= 1
            return httpx2.Response(500, json={"error": {"message": "server is busy"}})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)  # «думает»: одновременные запросы успевают наложиться
        self.in_flight -= 1
        if "prompt" in body:  # /v1/completions: ответ на каждый промпт пачки
            choices = [
                {"index": i, "text": f"answer {sum(ids)}", "finish_reason": "stop"}
                for i, ids in enumerate(body["prompt"])
            ]
            random.shuffle(choices)  # сервер вправе вернуть ответы не по порядку
            reply = {"object": "text_completion", "choices": choices}
        else:  # /v1/chat/completions: один диалог
            question = body["messages"][-1]["content"]
            message = {"role": "assistant", "content": "reply to " + question}
            reply = {
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            }
        return httpx2.Response(
            200, json={"id": "x", "created": 0, "model": body["model"], "usage": USAGE, **reply}
        )

    def http_client(self):
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))


def victim(server, **changes):
    options = {"http_client": server.http_client(), "retry_wait": 0}  # без пауз между повторами
    options.update(changes)
    return VictimClient("http://fake/v1", **options)


def test_victim_answers_come_in_prompt_order():
    server = FakeServer()
    prompts = [[i, i] for i in range(150)]  # у каждого промпта своя сумма ids
    answers = asyncio.run(victim(server).generate(prompts, GREEDY))
    assert answers == [f"answer {2 * i}" for i in range(150)]
    assert sorted(len(body["prompt"]) for body in server.bodies) == [22, 64, 64]  # пачки по 64


def test_victim_request_has_the_generation_params():
    server = FakeServer()
    asyncio.run(victim(server).generate([[5, 6]], GREEDY))
    body = server.bodies[0]
    assert body["model"] == "victim"
    assert body["prompt"] == [[5, 6]]  # готовые token ids, а не строка
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["max_tokens"] == 200
    assert body["seed"] == 0
    assert body["stop_token_ids"] == [1, 2, 3]


def test_victim_keeps_to_the_parallel_limit():
    server = FakeServer()
    client = victim(server, batch_size=1, parallel=3)
    asyncio.run(client.generate([[i] for i in range(20)], GREEDY))
    assert 2 <= server.max_in_flight <= 3  # запросы шли одновременно, но не больше трёх


def test_chat_answers_come_in_order_with_their_seeds():
    server = FakeServer()
    client = ChatClient("http://fake/v1", "attacker", parallel=4, http_client=server.http_client())
    convs = [[{"role": "user", "content": f"q{i}"}] for i in range(10)]
    seeds = [request_seed("run", i) for i in range(10)]
    params = GenParams(temperature=0.9, top_p=1.0, max_tokens=80, seed=0)
    answers = asyncio.run(client.chat(convs, params, seeds))
    assert answers == [f"reply to q{i}" for i in range(10)]
    seed_of = {body["messages"][-1]["content"]: body["seed"] for body in server.bodies}
    assert seed_of == {f"q{i}": seeds[i] for i in range(10)}  # у каждого диалога — свой сид
    assert "stop_token_ids" not in server.bodies[0]  # не заданы — не отправляются
    assert server.max_in_flight <= 4


def test_client_retries_server_errors():
    server = FakeServer(failures=2)  # два ответа 500, потом нормальный
    client = victim(server)
    assert asyncio.run(client.generate([[1]], GREEDY)) == ["answer 1"]
    assert len(server.bodies) == 3
    assert client.perf.retries == 2


def test_client_gives_up_after_max_attempts():
    server = FakeServer(failures=100)
    with pytest.raises(InternalServerError):
        asyncio.run(victim(server, max_attempts=5).generate([[1]], GREEDY))
    assert len(server.bodies) == 5


def test_perf_counts_requests_and_tokens():
    server = FakeServer()
    client = victim(server)
    asyncio.run(client.generate([[i] for i in range(130)], GREEDY))  # 3 пачки: 64 + 64 + 2
    summary = client.perf.summary()
    assert summary["requests"] == 3
    assert summary["prompt_tokens"] == 30  # по 10 за запрос (USAGE выше)
    assert summary["completion_tokens"] == 6  # по 2 за запрос
    assert summary["seconds_max"] > 0


def test_load_stop_token_ids(tmp_path):
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [128001, 128009]}')
    assert load_stop_token_ids(tmp_path) == [128001, 128009]
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 2}')
    assert load_stop_token_ids(tmp_path) == [2]


def test_request_seed_is_stable_and_differs_between_attempts():
    seed = request_seed("run", "forget01/0000", 5, 0)
    assert seed == request_seed("run", "forget01/0000", 5, 0)  # тот же адрес — тот же сид
    assert seed != request_seed("run", "forget01/0000", 5, 1)  # другая попытка — другой сид
    assert 0 <= seed < 2**31
