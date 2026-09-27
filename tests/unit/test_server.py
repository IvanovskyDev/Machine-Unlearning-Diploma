"""Тесты urec.serve.server без GPU: вместо vLLM — поддельная программа tests/unit/fake_vllm.py."""

import socket
import sys
from pathlib import Path

import pytest

from urec.serve.server import VLLMServer

FAKE_VLLM = [sys.executable, str(Path(__file__).parent / "fake_vllm.py")]


def free_port():
    """Свободный порт: если попросить порт 0, система выберет незанятый сама."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_server(tmp_path, **changes):
    """VLLMServer с поддельной программой и быстрыми проверками готовности."""
    options = {
        "model": "models/m",
        "name": "victim",
        "port": free_port(),
        "gpu_mem": 0.4,
        "max_model_len": 4096,
        "seed": 0,
        "log_dir": tmp_path,
        "command": FAKE_VLLM,
        "startup_timeout": 20,
        "poll_interval": 0.1,
    }
    options.update(changes)
    return VLLMServer(**options)


def test_command_line_follows_the_plan(tmp_path):
    server = make_server(tmp_path, command=["vllm"], port=8000)
    assert server.command_line() == [
        "vllm", "serve", "models/m",
        "--port", "8000",
        "--served-model-name", "victim",
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "4096",
        "--dtype", "bfloat16",
        "--generation-config", "vllm",
        "--enable-prefix-caching",
        "--seed", "0",
    ]  # fmt: skip


def test_server_starts_and_stops(tmp_path):
    with make_server(tmp_path) as server:
        process = server.process
        assert process.poll() is None  # сервер работает
        assert server.base_url == f"http://127.0.0.1:{server.port}/v1"
    assert process.poll() is not None  # после with — остановлен
    log = (tmp_path / "vllm_victim.log").read_text(encoding="utf-8")
    assert "fake vllm: serve models/m --port" in log  # вывод сервера — в лог


def test_server_stops_on_error_inside_with(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with make_server(tmp_path) as server:
            process = server.process
            1 / 0  # ошибка посреди работы
    assert process.poll() is not None  # сервер всё равно остановлен


def test_busy_port_is_an_error(tmp_path):
    with socket.socket() as busy:  # кто-то уже слушает порт
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        with pytest.raises(RuntimeError, match="занят"):
            with make_server(tmp_path, port=busy.getsockname()[1]):
                pass


def test_crashed_server_is_an_error_with_its_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VLLM_EXIT", "3")  # сервер сразу падает с кодом 3
    with pytest.raises(RuntimeError) as error:
        with make_server(tmp_path):
            pass
    assert "кодом 3" in str(error.value)
    assert "fake vllm: serve" in str(error.value)  # в сообщении — конец лога сервера


def test_slow_server_times_out_and_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VLLM_DELAY", "60")  # «грузится» дольше, чем мы готовы ждать
    server = make_server(tmp_path, startup_timeout=1)
    with pytest.raises(TimeoutError):
        with server:
            pass
    assert server.process is None  # stop() дождался завершения процесса
