"""Поддельная программа «vllm serve» для тестов VLLMServer: отвечает «200 OK» на /health.

Это не тест, а помощник: VLLMServer запускает её вместо настоящего vLLM.
Поведение задают переменные окружения:
- FAKE_VLLM_DELAY — сколько секунд «загружать модель», прежде чем отвечать;
- FAKE_VLLM_EXIT — сразу завершиться с этим кодом, как упавший сервер.
"""

import http.server
import os
import sys
import time

args = sys.argv[1:]  # serve <модель> --port … и остальные флаги
port = int(args[args.index("--port") + 1])
print("fake vllm:", " ".join(args), flush=True)  # попадёт в лог сервера

if os.environ.get("FAKE_VLLM_EXIT"):
    sys.exit(int(os.environ["FAKE_VLLM_EXIT"]))
time.sleep(float(os.environ.get("FAKE_VLLM_DELAY", "0")))


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):  # не писать в лог строку на каждый запрос
        pass


http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
