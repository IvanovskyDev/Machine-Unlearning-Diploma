"""Оркестратор (план, блоки 23, 26 и 37): шаги unlearn и eval форка по описанию запуска.

    python -m urec.pipeline run --spec results/raw/{run}/spec.yaml [ключ=значение ...]

Запуск описывает RunSpec (файл spec.yaml). Каждый шаг — команда форка в окружении unl:
её вывод идёт в лог logs/{run}/{шаг}.log, а выполненный шаг отмечается маркером DONE
в results/raw/{run}/{шаг}. Шаги с DONE пропускаются, поэтому повторный запуск
продолжает с упавшего шага. Что, как и на чём запускалось — в results/raw/{run}/manifest.json.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from urec.config import load_config
from urec.io import (
    DONE,
    atomic_write,
    is_done,
    mark_done,
    read_json,
    read_yaml,
    write_json,
    write_yaml,
)
from urec.types import Manifest, RunSpec, StepStatus

# forget-сплит → (retain-сплит, holdout-сплит). Тройки берутся только отсюда: holdout
# от другого сплита незаметно испортил бы privleak (план, блок 39)
SPLITS = {
    "forget01": ("retain99", "holdout01"),
    "forget05": ("retain95", "holdout05"),
    "forget10": ("retain90", "holdout10"),
}
# размер модели → имя её конфига в форке (external/open-unlearning/configs/model)
FORK_MODELS = {
    "1B": "Llama-3.2-1B-Instruct",
    "3B": "Llama-3.2-3B-Instruct",
    "8B": "Llama-3.1-8B-Instruct",
}
TARGET = "target"  # «метод» без забывания: шаг eval оценивает саму target-модель (Exp0)
DIRTY = "+dirty"  # пометка коммита, если в репозитории есть незакоммиченные правки
LOCK_FILES = ["requirements-unl.lock", "requirements-atk.lock"]  # lock-файлы окружений в envs/
LOG_TAIL_LINES = 30  # сколько последних строк лога показать при ошибке
# замер памяти GPU, как в scripts/measure.sh: раз в секунду — занятая память каждой GPU, МиБ
GPU_MONITOR = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-l", "1"]


class StepFailed(RuntimeError):
    """Шаг завершился с ошибкой. В сообщении — код завершения и конец лога шага."""

    def __init__(self, step: str, code: int, log: Path) -> None:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-LOG_TAIL_LINES:])
        super().__init__(f"шаг {step} завершился с кодом {code}. Конец лога {log}:\n{tail}")


@dataclass
class Step:
    """Один шаг запуска: программа форка, её добавки Hydra и что она должна создать."""

    name: str  # имя шага: "unlearn" или "eval"
    env: str  # окружение, в котором он идёт: "unl"
    script: str  # программа форка, например "src/train.py"
    overrides: list[str]  # добавки Hydra «ключ=значение»
    model: Path  # папка модели, которую шаг читает
    output: Path  # куда форк пишет результат (paths.output_dir)
    result: str  # файл в output, без которого шаг не считается выполненным


# --- Описание запуска ---


def run_name(exp: str, model_size: str, split: str, method: str, variant: str, seed: int) -> str:
    """Имя запуска {exp}_{model}_{split}_{method}_{variant}_s{seed}: exp2_3B_f05_NPO_base_s1."""
    short_split = "f" + split.removeprefix("forget")  # forget05 → f05
    return f"{exp}_{model_size}_{short_split}_{method}_{variant}_s{seed}"


def make_spec(
    exp: str,
    model_size: str,
    split: str,
    method: str,
    variant: str = "base",
    seed: int = 0,
    unlearn_overrides: Sequence[str] = (),
    keep_checkpoint: bool = False,
) -> RunSpec:
    """RunSpec для шагов unlearn и eval: имя и target-модель выводятся из параметров."""
    spec = RunSpec(
        run=run_name(exp, model_size, split, method, variant, seed),
        exp=exp,
        model_size=model_size,
        split=split,
        method=method,
        variant=variant,
        seed=seed,
        base_model=f"open-unlearning/tofu_{fork_model(model_size)}_full",  # M из TOFU
        unlearn_overrides=list(unlearn_overrides),
        regimes=[],  # поля атаки заполнит веха M3
        attackers=[],
        attack_seeds=[],
        retain_regime=None,
        keep_checkpoint=keep_checkpoint,
    )
    check_spec(spec)
    return spec


def check_spec(spec: RunSpec) -> None:
    """Проверяет, что сплит и размер модели известны, а имя запуска собрано из параметров."""
    fork_model(spec.model_size)  # ошибка, если размер неизвестен
    fork_splits(spec.split)  # ошибка, если сплит неизвестен
    expected = run_name(spec.exp, spec.model_size, spec.split, spec.method, spec.variant, spec.seed)
    if spec.run != expected:
        raise ValueError(f"имя запуска {spec.run} не по параметрам: должно быть {expected}")


def fork_model(model_size: str) -> str:
    """Имя конфига модели в форке по её размеру: "3B" → "Llama-3.2-3B-Instruct"."""
    if model_size not in FORK_MODELS:
        raise ValueError(f"неизвестный размер модели {model_size}; есть {', '.join(FORK_MODELS)}")
    return FORK_MODELS[model_size]


def fork_splits(split: str) -> tuple[str, str]:
    """(retain, holdout) для forget-сплита: "forget05" → ("retain95", "holdout05")."""
    if split not in SPLITS:
        raise ValueError(f"неизвестный сплит {split}; есть {', '.join(SPLITS)}")
    return SPLITS[split]


# --- Команды форка ---


def model_dir(cfg: DictConfig, model: str) -> Path:
    """Папка модели. HF id → {paths.models}/{имя после «/»}: так модели кладёт часть C."""
    if Path(model).is_absolute():  # уже путь к папке
        return Path(model)
    if cfg.paths.models is None:
        raise ValueError("не задана папка моделей: переменная окружения MODELS (часть C)")
    return Path(cfg.paths.models) / model.split("/")[-1]


def run_dir(cfg: DictConfig, spec: RunSpec) -> Path:
    """Папка запуска results/raw/{run}: spec, манифест, результаты шагов."""
    return Path(cfg.paths.raw) / spec.run


def checkpoint_dir(cfg: DictConfig, spec: RunSpec) -> Path:
    """Куда шаг unlearn сохраняет модель после забывания."""
    return Path(cfg.paths.checkpoints) / spec.run


def fork_overrides(cfg: DictConfig, spec: RunSpec) -> list[str]:
    """Добавки Hydra для обучения в форке (шаг unlearn) — RunSpec → команда (план, блок 23)."""
    retain, holdout = fork_splits(spec.split)
    model = fork_model(spec.model_size)
    base = model_dir(cfg, spec.base_model).as_posix()
    return [
        f"model={model}",
        f"model.model_args.pretrained_model_name_or_path={base}",
        f"model.tokenizer_args.pretrained_model_name_or_path={base}",  # не закрытый meta-llama
        f"forget_split={spec.split}",
        f"retain_split={retain}",
        f"holdout_split={holdout}",
        f"trainer={spec.method}",
        f"retain_logs_path=saves/eval/tofu_{model}_{retain}/TOFU_EVAL.json",  # оценка M_ret
        f"task_name={spec.run}",
        f"trainer.args.seed={spec.seed}",
        f"paths.output_dir={checkpoint_dir(cfg, spec).as_posix()}",
        *cfg.fork.unlearn_overrides,  # общие для всех запусков (configs/fork/default.yaml)
        *spec.unlearn_overrides,  # только для этого запуска
    ]


def eval_overrides(cfg: DictConfig, spec: RunSpec, model: Path, output: Path) -> list[str]:
    """Добавки Hydra для оценки модели model в форке (шаг eval); результат — в output."""
    retain, holdout = fork_splits(spec.split)
    name = fork_model(spec.model_size)
    base = model_dir(cfg, spec.base_model).as_posix()
    return [
        f"model={name}",
        f"model.model_args.pretrained_model_name_or_path={model.as_posix()}",
        f"model.tokenizer_args.pretrained_model_name_or_path={base}",  # токенизатор — всегда TOFU
        f"forget_split={spec.split}",
        f"holdout_split={holdout}",
        f"retain_logs_path=saves/eval/tofu_{name}_{retain}/TOFU_EVAL.json",
        f"task_name={spec.run}",
        "eval.tofu.overwrite=true",  # считать заново, даже если остался результат прошлой попытки
        f"paths.output_dir={output.as_posix()}",
    ]


def plan_steps(cfg: DictConfig, spec: RunSpec) -> list[Step]:
    """Шаги запуска по порядку: unlearn (если модель забывает) и eval."""
    check_spec(spec)
    base = model_dir(cfg, spec.base_model)
    if not (base / "config.json").exists():
        raise FileNotFoundError(f"модели {spec.base_model} нет в {base}: скачайте её (часть C)")
    steps: list[Step] = []
    model = base  # что оценивать: target-модель или чекпоинт после забывания
    if spec.method != TARGET:
        checkpoint = checkpoint_dir(cfg, spec)
        start = ["--config-name=unlearn.yaml", "experiment=unlearn/tofu/default"]
        unlearn = Step(
            name="unlearn",
            env="unl",
            script="src/train.py",
            overrides=start + fork_overrides(cfg, spec),
            model=base,
            output=checkpoint,
            result="config.json",  # есть у каждой сохранённой модели
        )
        steps.append(unlearn)
        model = checkpoint
    output = run_dir(cfg, spec) / "eval"
    start = ["--config-name=eval.yaml", "experiment=eval/tofu/default"]
    evaluate = Step(
        name="eval",
        env="unl",
        script="src/eval.py",
        overrides=start + eval_overrides(cfg, spec, model, output),
        model=model,
        output=output,
        result="TOFU_SUMMARY.json",  # итоговые метрики TOFU
    )
    steps.append(evaluate)
    return steps


# --- Манифест (блок 37) ---


def git_commit(folder: Path) -> str:
    """Коммит репозитория в папке; +dirty — если в отслеживаемых файлах есть правки."""
    commit = _git(folder, "rev-parse", "HEAD")
    # --untracked-files=no: новые файлы (логи, результаты) код не меняют и не считаются
    changes = _git(folder, "status", "--porcelain", "--untracked-files=no")
    return commit + DIRTY if changes else commit


def lock_sha(root: Path) -> str:
    """Общий отпечаток sha256 lock-файлов окружений: меняется, если изменился любой из них."""
    digest = hashlib.sha256()
    for name in LOCK_FILES:
        digest.update((root / "envs" / name).read_bytes())
    return digest.hexdigest()


def model_revisions(root: Path) -> dict[str, str]:
    """Ревизии моделей из envs/models.lock.json (часть C)."""
    text = (root / "envs" / "models.lock.json").read_text(encoding="utf-8")
    revisions: dict[str, str] = json.loads(text)
    return revisions


def hardware() -> str:
    """GPU, версия драйвера и CUDA по данным nvidia-smi."""
    gpus = _nvidia_smi("--query-gpu=name,driver_version", "--format=csv,noheader")
    if gpus is None:
        return "GPU нет: nvidia-smi не найден или не работает"
    names = []
    for line in gpus.strip().splitlines():  # по строке на каждую GPU: «имя, драйвер»
        name, driver = line.rsplit(",", 1)  # по последней запятой: имя тоже может её содержать
        names.append(f"{name.strip()}, драйвер {driver.strip()}")
    # версия CUDA — из шапки nvidia-smi: «CUDA Version: 12.4», у новых драйверов «CUDA UMD Version: 13.3»
    cuda = re.search(r"CUDA (?:UMD )?Version: ([\d.]+)", _nvidia_smi() or "")
    return f"{'; '.join(names)}; CUDA {cuda.group(1) if cuda else '?'}"


def check_git(cfg: DictConfig) -> dict[str, str]:
    """Коммиты urec и форка. Незакоммиченные правки — ошибка, если не задано allow_dirty=true."""
    git = {"urec": git_commit(Path(cfg.paths.root)), "fork": git_commit(Path(cfg.paths.fork_dir))}
    dirty = [name for name, commit in git.items() if commit.endswith(DIRTY)]
    if dirty and not cfg.allow_dirty:
        raise RuntimeError(
            f"незакоммиченные правки в {', '.join(dirty)}: "
            "закоммитьте их или запустите с allow_dirty=true (план, блок 37)"
        )
    return git


def start_manifest(cfg: DictConfig, spec: RunSpec, git: dict[str, str]) -> Manifest:
    """Пишет манифест запуска: коммиты, lock-файлы, модели, железо; шаги — из прошлого запуска."""
    root = Path(cfg.paths.root)
    path = run_dir(cfg, spec) / "manifest.json"
    steps: dict[str, StepStatus] = {}
    if path.exists():  # запуск продолжается: история шагов сохраняется
        old = read_json(path, Manifest)
        steps = old.steps
        if old.git != git:
            print(f"внимание: код изменился с прошлого запуска: {old.git} → {git}", flush=True)
    manifest = Manifest(
        run=spec.run,
        git=git,
        lock_sha=lock_sha(root),
        model_revisions=model_revisions(root),
        hardware=hardware(),
        steps=steps,
    )
    write_json(path, manifest)
    return manifest


# --- Запуск ---


def run(cfg: DictConfig, spec: RunSpec) -> Manifest:
    """Выполняет шаги запуска по порядку; выполненные раньше (с DONE) пропускает."""
    steps = plan_steps(cfg, spec)  # сначала всё проверить, потом писать файлы и запускать
    git = check_git(cfg)
    folder = run_dir(cfg, spec)
    save_spec(folder / "spec.yaml", spec)  # до манифеста: чужую папку запуска не трогать
    config_text = OmegaConf.to_yaml(cfg, resolve=True)  # снимок конфига urec (блок 37)
    atomic_write(folder / "config_snapshot.yaml", config_text.encode("utf-8"))
    manifest = start_manifest(cfg, spec, git)
    for step in steps:
        run_step(cfg, spec, step, manifest)
    return manifest


def save_spec(path: Path, spec: RunSpec) -> None:
    """Кладёт spec.yaml в папку запуска. Другой spec под тем же именем — ошибка."""
    if path.exists():
        if read_yaml(path, RunSpec) != spec:
            raise ValueError(f"в {path} уже другой spec: имя {spec.run} занято другим запуском")
        return
    write_yaml(path, spec)


def run_step(cfg: DictConfig, spec: RunSpec, step: Step, manifest: Manifest) -> None:
    """Выполняет шаг, если он ещё не выполнен: лог, замер памяти GPU, манифест, маркер DONE."""
    out = run_dir(cfg, spec) / step.name  # папка маркера DONE и снимка конфига форка
    if is_done(out):
        print(f"{step.name}: уже выполнен, пропускаю", flush=True)
        return
    if not (step.model / "config.json").exists():
        raise FileNotFoundError(
            f"шагу {step.name} нужна модель {step.model}, а её нет. Если это чекпоинт, удалите "
            f"{run_dir(cfg, spec) / 'unlearn' / DONE}: тогда шаг unlearn выполнится заново"
        )
    command = [cfg.envs[step.env], step.script, *step.overrides]
    log = Path(cfg.paths.logs) / spec.run / f"{step.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status = StepStatus(
        started=_now(),
        finished=None,
        status="running",
        seconds=None,
        command=command,
        peak_gpu_mib=None,
    )
    manifest.steps[step.name] = status
    manifest_path = run_dir(cfg, spec) / "manifest.json"
    write_json(manifest_path, manifest)  # в манифесте видно, какой шаг идёт сейчас
    print(f"{step.name}: запуск, лог — {log}", flush=True)

    gpu_log = log.with_name(f"{step.name}_gpu.log")
    monitor = _start_gpu_monitor(gpu_log)
    start = time.monotonic()
    status.status = "failed"  # станет "done", только если шаг выполнится целиком
    try:
        code = _execute(command, cfg.paths.fork_dir, log, status.started)
        if code == 0:
            _collect_result(step, out)
            status.status = "done"
    finally:  # и при успехе, и при ошибке: время, память и итог — в манифест
        status.peak_gpu_mib = _stop_gpu_monitor(monitor, gpu_log)
        status.seconds = round(time.monotonic() - start, 1)
        status.finished = _now()
        write_json(manifest_path, manifest)
    if code != 0:
        raise StepFailed(step.name, code, log)
    mark_done(out)  # последним: шаг выполнен целиком
    print(f"{step.name}: готово за {status.seconds:.0f} с", flush=True)


def _execute(command: list[str], cwd: str, log: Path, started: str) -> int:
    """Запускает команду в папке cwd, дописывая её вывод в лог. Возвращает код завершения."""
    python_dir = str(Path(command[0]).parent)  # папка python окружения
    env = dict(os.environ)
    env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")  # как после activate
    env["PYTHONUNBUFFERED"] = "1"  # вывод попадает в лог сразу, а не пачками
    env["PYTHONIOENCODING"] = "utf-8"  # вывод — в UTF-8, как и заголовок лога
    with open(log, "a", encoding="utf-8") as f:  # "a": лог прошлых попыток сохраняется
        f.write(f"\n=== {started}: {shlex.join(command)}\n")  # когда и что запущено
        f.flush()  # заголовок — в файл раньше вывода программы
        result = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return result.returncode


def _collect_result(step: Step, out: Path) -> None:
    """Проверяет, что шаг создал свой результат, и копирует в out снимок конфига форка."""
    result = step.output / step.result
    if not result.exists():
        raise RuntimeError(f"шаг {step.name} завершился без ошибки, но не создал {result}")
    out.mkdir(parents=True, exist_ok=True)
    # Hydra форка сохраняет итоговый конфиг шага в .hydra/config.yaml (план, блок 37)
    shutil.copy(step.output / ".hydra" / "config.yaml", out / "fork_config.yaml")


def _start_gpu_monitor(path: Path) -> subprocess.Popen[bytes] | None:
    """Запускает в фоне замер памяти GPU в файл path; без GPU — None."""
    if shutil.which(GPU_MONITOR[0]) is None:  # nvidia-smi нет: машина без GPU
        return None
    with open(path, "wb") as f:
        return subprocess.Popen(GPU_MONITOR, stdout=f, stderr=subprocess.DEVNULL)


def _stop_gpu_monitor(monitor: subprocess.Popen[bytes] | None, path: Path) -> int | None:
    """Останавливает замер и возвращает пик занятой памяти GPU, МиБ."""
    if monitor is None:
        return None
    monitor.terminate()
    monitor.wait()
    values = [int(word) for word in path.read_text().split() if word.isdigit()]
    return max(values, default=None)


def _git(folder: Path, *args: str) -> str:
    """Выполняет команду git в папке folder и возвращает её вывод."""
    result = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} в {folder}: {result.stderr.strip()}")
    return result.stdout.strip()


def _nvidia_smi(*args: str) -> str | None:
    """Вывод nvidia-smi с аргументами args; None, если программы нет или она не работает."""
    try:
        result = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout if result.returncode == 0 else None


def _now() -> str:
    """Текущее время с часовым поясом, например 2026-10-14T12:30:05+03:00."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main(argv: Sequence[str] | None = None) -> None:
    """Разбирает командную строку и выполняет запуск."""
    parser = argparse.ArgumentParser(prog="python -m urec.pipeline", description=__doc__)
    parser.add_argument("command", choices=["run"], help="run — выполнить шаги запуска")
    parser.add_argument("--spec", required=True, help="файл spec.yaml запуска")
    parser.add_argument("overrides", nargs="*", help="значения конфига urec: ключ=значение")
    args = parser.parse_intermixed_args(argv)  # --spec можно писать до или после добавок
    cfg = load_config(args.overrides)
    spec = read_yaml(args.spec, RunSpec)
    try:
        run(cfg, spec)
    except StepFailed as error:
        raise SystemExit(str(error)) from None  # понятное сообщение без traceback
    print(f"{spec.run}: все шаги выполнены", flush=True)


if __name__ == "__main__":  # файл запущен как программа: python -m urec.pipeline …
    main()
