"""Сервер vLLM (план, блок 26): модель в отдельном процессе, который гасится при любом исходе.

    with VLLMServer(model=ckpt, name="victim", port=8000, gpu_mem=0.40,
                    max_model_len=4096, seed=0, log_dir=logs) as victim:
        client = VictimClient(victim.base_url)
        ...

Выход из with — даже из-за ошибки — останавливает сервер и освобождает память GPU.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import IO

LOG_TAIL_LINES = 30  # сколько последних строк лога показать при ошибке


class VLLMServer:
    """Сервер vLLM с одной моделью: запускается при входе в with, гасится при выходе."""

    def __init__(
        self,
        model: str | Path,
        name: str,
        port: int,
        gpu_mem: float,
        max_model_len: int,
        seed: int,
        log_dir: str | Path,
        dtype: str = "bfloat16",
        extra_args: Sequence[str] = (),
        command: Sequence[str] | None = None,
        startup_timeout: float = 900.0,
        stop_timeout: float = 30.0,
        poll_interval: float = 5.0,
    ) -> None:
        self.model = str(model)  # папка модели или её id на Hugging Face
        self.name = name  # имя модели на сервере: так к ней обращаются клиенты
        self.port = port
        self.gpu_mem = gpu_mem  # доля памяти GPU для сервера
        self.max_model_len = max_model_len  # самый длинный промпт вместе с ответом, в токенах
        self.seed = seed
        self.log_path = Path(log_dir) / f"vllm_{name}.log"
        self.dtype = dtype  # bfloat16 — как в оценке; модели во float32 тоже считаются в bf16
        self.extra_args = list(extra_args)  # другие флаги vllm serve, если понадобятся
        # программа vllm лежит в той же папке окружения, что и python (в Colab — /content/envs/atk/bin)
        self.command = list(command) if command else [str(Path(sys.executable).with_name("vllm"))]
        self.startup_timeout = startup_timeout  # сколько ждать готовности; модель грузится минутами
        self.stop_timeout = stop_timeout  # сколько ждать мягкой остановки, потом — принудительная
        self.poll_interval = poll_interval  # как часто спрашивать сервер, готов ли он
        self.process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    @property
    def base_url(self) -> str:
        """Адрес для клиентов (VictimClient, ChatClient)."""
        return f"http://127.0.0.1:{self.port}/v1"

    def command_line(self) -> list[str]:
        """Команда запуска сервера (план, блок 26)."""
        return [
            *self.command,
            "serve",
            self.model,
            "--port", str(self.port),
            "--served-model-name", self.name,
            "--gpu-memory-utilization", str(self.gpu_mem),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--generation-config", "vllm",  # не брать temperature и др. из generation_config.json
            "--enable-prefix-caching",  # общие начала промптов считаются один раз
            "--seed", str(self.seed),
            *self.extra_args,
        ]  # fmt: skip

    def __enter__(self) -> "VLLMServer":
        if _port_is_busy(self.port):
            raise RuntimeError(f"порт {self.port} занят: другой сервер ещё работает?")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")  # весь вывод сервера — в лог
        try:
            self.process = subprocess.Popen(
                self.command_line(),
                stdout=self._log,
                stderr=subprocess.STDOUT,
                # своя группа процессов: так при остановке гасятся и дочерние процессы vLLM
                start_new_session=(sys.platform != "win32"),
            )
            self._wait_until_ready()
        except BaseException:  # не запустился или не дождались — погасить и закрыть лог
            self.stop()
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()  # и при обычном выходе, и при ошибке внутри with

    def _wait_until_ready(self) -> None:
        """Ждёт, пока сервер ответит на /health; упал или не успел — ошибка с концом лога."""
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            assert self.process is not None
            if self.process.poll() is not None:  # процесс сервера уже завершился
                raise RuntimeError(
                    f"сервер {self.name} завершился с кодом {self.process.returncode}. "
                    f"Конец лога {self.log_path}:\n{self._log_tail()}"
                )
            if _is_healthy(self.port):
                return
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"сервер {self.name} не ответил за {self.startup_timeout:.0f} с. "
            f"Конец лога {self.log_path}:\n{self._log_tail()}"
        )

    def stop(self) -> None:
        """Останавливает сервер: сначала мягко, через stop_timeout секунд — принудительно."""
        if self.process is not None:
            if self.process.poll() is None:
                self._signal_all(kill=False)  # просьба завершиться
                try:
                    self.process.wait(timeout=self.stop_timeout)
                except subprocess.TimeoutExpired:
                    pass
            self._signal_all(kill=True)  # добить всё, что осталось, в том числе дочерние процессы
            self.process.wait()
            self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def _signal_all(self, kill: bool) -> None:
        """Сигнал серверу и его дочерним процессам: SIGTERM (мягко) или SIGKILL (принудительно)."""
        assert self.process is not None
        if sys.platform == "win32":  # на Windows (только в тестах) групп процессов нет
            if self.process.poll() is None:
                if kill:
                    self.process.kill()
                else:
                    self.process.terminate()
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL if kill else signal.SIGTERM)
        except ProcessLookupError:  # в группе уже никого нет
            pass

    def _log_tail(self) -> str:
        """Последние строки лога сервера."""
        if self._log is not None:
            self._log.flush()
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-LOG_TAIL_LINES:])


def _port_is_busy(port: int) -> bool:
    """Отвечает ли уже кто-то на этом порту."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _is_healthy(port: int) -> bool:
    """Отвечает ли сервер «200 OK» на запрос /health."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            status: int = response.status  # код ответа: 200 — всё хорошо
    except (urllib.error.URLError, OSError):  # ещё не слушает порт или не готов
        return False
    return status == 200
