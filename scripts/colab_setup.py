"""Настройка Colab для блокнотов диплома: Drive, git, uv, окружения unl и atk, журнал.

Всё, что раньше делали ячейки %%bash, теперь делают функции этого файла, а блокноты только
вызывают их: colab_setup.start_session(), colab_setup.build_envs() и так далее. Команды
терминала (git, uv, nvidia-smi) запускает функция run() через модуль subprocess.

Файл подключает первая ячейка каждого блокнота: она клонирует репозиторий в /content/repo
и добавляет папку scripts в sys.path, после чего работает `import colab_setup`.
"""

import os
import re
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import psutil

# google.colab и huggingface_hub импортируются внутри функций, а не здесь. google.colab есть
# только в самом Colab. А huggingface_hub запоминает папку кэша HF_HOME в момент импорта,
# поэтому импортировать его можно только после того, как start_session() задаст HF_HOME.


# ---------------------------------------------------------------------------------------------
# Где что лежит. Машина Colab временная, поэтому у каждой папки заранее определена роль.
# ---------------------------------------------------------------------------------------------

# Google Drive: всё, что нельзя потерять, — чекпоинты, результаты атак, логи, журнал
DRIVE = "/content/drive/MyDrive/unlearning_data"
# диск машины: кэш и модели; стираются после сессии, но скачиваются заново за минуты
FAST = "/content/fast"
# окружения Python unl и atk; собираются заново в каждой сессии
ENVS = "/content/envs"
# репозиторий диплома; его клонирует первая ячейка блокнота
REPO = "/content/repo"
# форк OpenUnlearning — submodule репозитория; правка ошибки bfloat16 уже в его коде
FORK = REPO + "/external/open-unlearning"
# лабораторный журнал: где, чем и с каким результатом запускали (пригодится для главы 3)
JOURNAL = DRIVE + "/journal.md"

# папки, которые нужны блокнотам
FOLDERS = [
    DRIVE + "/saves",  # чекпоинты и оценки OpenUnlearning
    DRIVE + "/results_raw",  # сырые результаты атак
    DRIVE + "/logs",  # логи запусков, замеры времени и памяти
    DRIVE + "/data",  # таблицы и ответы моделей для чтения глазами
    DRIVE + "/envs",  # models.lock.json и копии lock-файлов (части C и D)
    FAST + "/hf_home",  # кэш Hugging Face
    FAST + "/models",  # скачанные модели
]

# Готовая сборка FlashAttention 2.6.3 под Python 3.11 (cp311), torch 2.4 и CUDA 12 (cu123):
# с ней не нужна часовая компиляция. В lock-файл unl она не входит и ставится отдельно.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)


# ---------------------------------------------------------------------------------------------
# Запуск команд терминала
# ---------------------------------------------------------------------------------------------


def command_env(env_name=None):
    """Переменные окружения для запускаемой команды.

    Берём переменные блокнота и убираем три настройки Colab, которые ломают наши окружения:
      UV_...     — велят uv ставить пакеты в системный Python Colab с его ограничениями версий;
      PYTHONPATH — подмешивает модули Colab в любой запущенный Python;
      MPLBACKEND — настройка графиков блокнота; с ней в наших окружениях падают vLLM и BERTScore.
    Меняем только копию: самому блокноту эти настройки нужны (с MPLBACKEND графики рисуются
    прямо в ячейках).

    env_name — "unl" или "atk": тогда команда работает в этом окружении так же, как после
    `source /content/envs/<имя>/bin/activate` — папка bin окружения идёт первой в PATH,
    а VIRTUAL_ENV подсказывает uv, куда ставить пакеты.
    """
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("UV_") or name in ["PYTHONPATH", "MPLBACKEND"]:
            del env[name]
    if env_name is not None:
        env["VIRTUAL_ENV"] = ENVS + "/" + env_name
        env["PATH"] = ENVS + "/" + env_name + "/bin:" + env["PATH"]
    return env


def run(command, env_name=None, cwd=None):
    """Выполнить команду терминала и показывать её вывод в ячейке, пока она работает.

    command  — список слов команды, например ["uv", "pip", "check"];
    env_name — "unl" или "atk", если команда должна работать в нашем окружении;
    cwd      — папка, в которой выполнить команду (как cd перед ней).
    Если команда завершилась с ошибкой, run() останавливает ячейку — как `set -e` в bash.
    """
    print("$", " ".join(command))  # как в терминале: видно, какая команда сейчас работает
    # Вывод команды забираем себе (PIPE) и печатаем сами: иначе Colab может его не показать.
    # stderr=STDOUT — сообщения об ошибках идут в тот же поток, вперемешку с обычным выводом.
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=command_env(env_name),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Печатаем строку за строкой, как только она появилась, — так виден ход долгих установок.
    # Строки приходят байтами, и в текст их переводим сами: если бы это делал Python (text=True),
    # каждое обновление полоски прогресса стало бы отдельной строкой.
    # errors="replace" — не падать, если в выводе попадётся битый символ.
    for line in process.stdout:
        print(line.decode("utf-8", errors="replace"), end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Команда завершилась с ошибкой (код {process.returncode}): {command}")


def get_output(command, env_name=None, cwd=None):
    """Выполнить команду и вернуть её вывод строкой — для команд, чей ответ нужен в коде."""
    result = subprocess.run(
        command, cwd=cwd, env=command_env(env_name), capture_output=True, text=True, check=True
    )
    return result.stdout


def commit(folder):
    """Короткий номер последнего коммита git в папке: по нему видно, какой код работал."""
    return get_output(["git", "-C", folder, "log", "-1", "--format=%h"]).strip()


# ---------------------------------------------------------------------------------------------
# Начало и конец сессии Colab
# ---------------------------------------------------------------------------------------------


def start_session():
    """Начало каждой сессии: Drive, папки, ссылки на Drive, переменные окружения, токен HF."""
    from google.colab import drive, userdata

    # Drive — первым делом: всё важное хранится там, ведь машина Colab стирается после сессии.
    # Colab попросит разрешение на доступ — согласитесь
    drive.mount("/content/drive")
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)  # exist_ok=True — если папка уже есть, не ругаться

    # Ссылки из репозитория на Drive: программы пишут «к себе в папку», а файлы сразу попадают
    # на Drive. В git ссылки не попадают — они записаны в .gitignore репозитория и форка
    make_link(DRIVE + "/saves", FORK + "/saves")  # чекпоинты и оценки OpenUnlearning
    make_link(DRIVE + "/results_raw", REPO + "/results/raw")  # сырые результаты атак

    # Переменные окружения видят все программы, которые запускает блокнот
    os.environ["HF_HOME"] = FAST + "/hf_home"  # куда Hugging Face складывает скачанное
    os.environ["MODELS"] = FAST + "/models"  # папка моделей; её читает configs/paths/default.yaml
    os.environ["BIG"] = DRIVE  # «большой диск» из плана; его ждут scripts/*.sh
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # меньше лишних предупреждений токенизаторов
    # Python-программы печатают сразу, а не пачками, — иначе run() показывал бы вывод с опозданием
    os.environ["PYTHONUNBUFFERED"] = "1"
    # токен Hugging Face — из секрета HF_TOKEN (🔑 слева в Colab), поэтому в коде его нет
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

    from huggingface_hub import whoami  # только теперь: HF_HOME уже задан

    print("Drive:", DRIVE)
    print("Hugging Face:", whoami()["name"])  # имя владельца токена — значит, токен работает
    print("Код: репозиторий", commit(REPO), "| форк OpenUnlearning", commit(FORK))


def make_link(target, link):
    """Создать ссылку link → target, если её ещё нет, и показать, куда она ведёт."""
    # Если на месте ссылки окажется обычная папка, os.symlink упадёт с ошибкой — и это хорошо:
    # иначе результаты молча писались бы на диск машины и пропали бы вместе с ней
    if not os.path.islink(link):
        os.symlink(target, link)
    print(link, "→", os.readlink(link))


def finish_session():
    """Конец сессии: дописать на Drive всё, что ещё не записано, и отключить его."""
    from google.colab import drive

    drive.flush_and_unmount()
    print("Все файлы записаны на Drive.")
    print("Машину можно отключить: Runtime → Disconnect and delete runtime")


# ---------------------------------------------------------------------------------------------
# Машина
# ---------------------------------------------------------------------------------------------


def machine_info():
    """Сводка о машине Colab — для экрана и для журнала: GPU, CUDA, процессор, память, диск."""
    # имя GPU, её память, compute capability и версия драйвера — одной строкой через запятую
    query = "--query-gpu=name,memory.total,compute_cap,driver_version"
    gpu = get_output(["nvidia-smi", query, "--format=csv,noheader"]).splitlines()[0]
    # версию CUDA nvidia-smi пишет только в шапке своей таблицы — ищем её регулярным выражением
    cuda = re.search(r"CUDA Version: [0-9.]+", get_output(["nvidia-smi"])).group()
    ram = round(psutil.virtual_memory().total / 1e9)  # ОЗУ в гигабайтах (1e9 — миллиард байт)
    disk = round(shutil.disk_usage("/content").free / 1e9)  # свободно на диске машины, ГБ
    return (
        f"- GPU (имя, память, compute capability, драйвер): {gpu}\n"
        f"- {cuda}\n"
        f"- CPU: {os.cpu_count()} ядер, ОЗУ: {ram} ГБ, свободно на диске машины: {disk} ГБ\n"
        f"- Большой диск: {DRIVE}\n"
    )


# ---------------------------------------------------------------------------------------------
# Окружения Python: unl и atk
# ---------------------------------------------------------------------------------------------
# Окружение — отдельная папка со своим Python и своими пакетами. Окружений два, потому что им
# нужны несовместимые версии одних и тех же библиотек: OpenUnlearning проверен с torch 2.4.1
# и transformers 4.51.3, а vLLM 0.19.1 требует torch 2.10 и transformers 5.
#   unl — обучение и оценка моделей кодом форка OpenUnlearning;
#   atk — атаки и анализ: vLLM, spaCy, BERTScore и наш пакет urec.
# Точные версии всех пакетов записаны в lock-файлах envs/ репозитория: по ним окружения
# собираются одинаково в каждой сессии, поэтому числа в дипломе воспроизводятся.


def build_envs():
    """Собрать окружения unl и atk ровно по lock-файлам репозитория. Несколько минут."""
    install_uv()

    create_venv("unl")
    run(["uv", "pip", "install", "-r", REPO + "/envs/requirements-unl.lock"], env_name="unl")
    run(["uv", "pip", "install", FLASH_ATTN_WHEEL], env_name="unl")

    create_venv("atk")
    run(["uv", "pip", "install", "-r", REPO + "/envs/requirements-atk.lock"], env_name="atk")
    # urec ставим «в режиме разработки» (-e): в окружение попадает ссылка на папку src/urec,
    # а не копия кода. Всё, что нужно urec, уже есть в lock-файле, так что atk не меняется
    run(["uv", "pip", "install", "-e", REPO], env_name="atk")
    run(["uv", "pip", "check"], env_name="atk")  # версии всех пакетов совместимы между собой


def install_uv():
    """Поставить uv — установщик пакетов вместо conda: он быстро собирает окружения."""
    run(["pip", "install", "-q", "uv"])  # -q — без подробного вывода
    run(["uv", "--version"])


def create_venv(name):
    """Создать пустое окружение /content/envs/<name> с Python 3.11."""
    # --python 3.11 — uv сам скачает этот Python: у Colab свой, более новый;
    # --seed — сразу положить в окружение pip, setuptools и wheel (они есть и в lock-файлах);
    # --clear — если окружение уже есть, стереть его и собрать заново
    run(["uv", "venv", ENVS + "/" + name, "--python", "3.11", "--seed", "--clear"])


def check_envs():
    """Самопроверка обоих окружений: по каждой проверке — OK или FAIL. Около трёх минут.

    Сами проверки — в scripts/check_env.py: его запускает python проверяемого окружения,
    потому что только так видны пакеты этого окружения (vLLM, spaCy, FlashAttention…).
    """
    failed = []
    for env_name in ["unl", "atk"]:
        try:
            run(["python", REPO + "/scripts/check_env.py", env_name], env_name=env_name)
        except RuntimeError:  # check_env.py завершается с ошибкой, если хоть одна проверка — FAIL
            failed.append(env_name)
    if failed:
        raise RuntimeError("Не все проверки прошли, окружения: " + ", ".join(failed))
    print("OK: все проверки прошли")


# ---------------------------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------------------------


def download_eval_logs():
    """Скачать оценки готовых моделей TOFU от авторов OpenUnlearning в saves/eval на Drive.

    По ним считается Forget Quality. Файлы лежат на Drive, поэтому хватает одного раза на весь
    проект. Это тот же вызов, что делает `python setup_data.py --eval_logs` в папке форка.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="open-unlearning/eval",
        repo_type="dataset",
        allow_patterns="*.json",
        local_dir=DRIVE + "/saves/eval",
    )
    for name in sorted(os.listdir(DRIVE + "/saves/eval")):
        if not name.startswith("."):  # служебную папку .cache не показываем
            print("   ", name)


# ---------------------------------------------------------------------------------------------
# Лабораторный журнал
# ---------------------------------------------------------------------------------------------


def write_journal(title, text):
    """Дописать в журнал на Drive запись с сегодняшней датой и показать её на экране."""
    today = datetime.now(ZoneInfo("Europe/Moscow")).date()  # по Москве: часы Colab идут по UTC
    entry = f"\n## {today} — {title}\n{text}"
    with open(JOURNAL, "a", encoding="utf-8") as f:  # "a" — дописать в конец файла
        f.write(entry)
    print(entry)


def envs_info():
    """Сводка об окружениях для журнала: версии кода и главных пакетов из lock-файлов."""
    unl = versions_from_lock("unl", ["torch", "transformers", "accelerate", "deepspeed"])
    atk = versions_from_lock("atk", ["torch", "vllm", "transformers", "spacy", "cupy-cuda12x"])
    return (
        f"- Код: репозиторий {commit(REPO)}, форк OpenUnlearning {commit(FORK)}\n"
        f"- unl: Python 3.11, {unl}, flash-attn 2.6.3\n"
        f"- atk: Python 3.11, {atk}, пакет urec\n"
        "- Lock-файлы: envs/requirements-unl.lock и envs/requirements-atk.lock репозитория\n"
    )


def versions_from_lock(env_name, names):
    """Строки «пакет==версия» из lock-файла окружения для пакетов из списка names."""
    found = []
    with open(REPO + "/envs/requirements-" + env_name + ".lock", encoding="utf-8") as f:
        for line in f:
            if line.split("==")[0] in names:  # "torch==2.4.1" → "torch"
                found.append(line.strip())
    return ", ".join(found)


# ---------------------------------------------------------------------------------------------
# Приложение: сборка окружений с нуля — только когда нужно сменить версии пакетов
# ---------------------------------------------------------------------------------------------
# Обычная сессия собирает окружения по lock-файлам (build_envs), поэтому версии сами не
# меняются. Чтобы их обновить, поправьте списки ниже и в новой сессии выполните
# build_envs_from_scratch(): новые lock-файлы появятся в envs/ на Drive. Скопируйте их в envs/
# репозитория на Windows, закоммитьте и прогоните блокноты заново.

# unl: OpenUnlearning со всеми зависимостями и lm-eval (для MMLU) — «.» здесь папка форка;
# setuptools<81 — в новых версиях нет модуля pkg_resources, без него не запускается TensorBoard
UNL_PACKAGES = [".[lm-eval]", "setuptools<81"]

# модель spaCy en_core_web_trf 3.8.0 — та же, что ставит команда spacy download
SPACY_MODEL = (
    "en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"
)

ATK_PACKAGES = [
    # vLLM 0.19.1 — последняя версия под CUDA 12: работает с драйверами и CUDA 12, и CUDA 13.
    # transformers, tokenizers и huggingface-hub — ровно те версии, с которыми тестировался
    # vLLM 0.19.1; setuptools<81 — как в тестах vLLM: в новых версиях нет pkg_resources
    "vllm==0.19.1",
    "transformers==5.5.3",
    "tokenizers==0.22.2",
    "huggingface-hub==1.10.2",
    "setuptools<81",
    # клиент к серверу vLLM и оценка ответов: сущности, близость по смыслу, нечёткий поиск
    "openai",
    "spacy",
    # spaCy на GPU. Отдельным пакетом, а не через spacy[cuda12x]: иначе uv откатит vLLM
    "cupy-cuda12x",
    "bert-score",
    "sentence-transformers",
    "rapidfuzz",
    # данные, конфиги, повторы запросов
    "pandas",
    "pyarrow",
    "orjson",
    "zstandard",
    "hydra-core",
    "tenacity",
    # статистика и графики
    "scipy",
    "statsmodels",
    "scikit-learn",
    "lifelines",
    "matplotlib",
    # тесты и проверки кода
    "pytest",
    "hypothesis",
    "ruff",
    "mypy",
    SPACY_MODEL,
]


def build_envs_from_scratch():
    """Собрать unl и atk по спискам выше и записать новые lock-файлы в envs/ на Drive."""
    install_uv()

    create_venv("unl")
    run(["uv", "pip", "install"] + UNL_PACKAGES, env_name="unl", cwd=FORK)
    run(["uv", "pip", "install", FLASH_ATTN_WHEEL], env_name="unl")
    save_lock("unl")

    create_venv("atk")
    run(["uv", "pip", "install"] + ATK_PACKAGES, env_name="atk")
    run(["uv", "pip", "install", "-e", REPO], env_name="atk")
    run(["uv", "pip", "check"], env_name="atk")
    save_lock("atk")


def save_lock(env_name):
    """Записать точные версии всех пакетов окружения в envs/requirements-<имя>.lock на Drive."""
    # --exclude-editable — без urec: он ставится из папки репозитория, а не по lock-файлу
    freeze = get_output(["uv", "pip", "freeze", "--exclude-editable"], env_name=env_name)
    # FlashAttention и сам OpenUnlearning тоже ставятся отдельно, поэтому в lock-файл не идут
    # (то же правило — в scripts/check_env.py, когда он сверяет окружение с lock-файлом)
    lines = []
    for line in freeze.splitlines():
        if "flash-attn" not in line and "open-unlearning" not in line:
            lines.append(line)
    path = DRIVE + "/envs/requirements-" + env_name + ".lock"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(path, "—", len(lines), "пакетов")
