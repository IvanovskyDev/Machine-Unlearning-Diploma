"""Тесты urec.pipeline без GPU: вместо форка — поддельный форк tests/unit/fake_fork.

Фикстура project строит маленький проект во временной папке: репозиторий urec
с lock-файлами, поддельный форк (оба — настоящие репозитории git) и папку модели.
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from urec import pipeline
from urec.config import load_config
from urec.io import is_done, read_json, read_yaml, write_yaml
from urec.types import Manifest, RunSpec

FAKE_FORK = Path(__file__).parent / "fake_fork"
BASE_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"


def git_init(folder):
    """Делает папку репозиторием git с одним коммитом всех файлов."""

    def git(*args):
        subprocess.run(["git", "-C", str(folder), *args], check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "init")


@pytest.fixture
def project(tmp_path):
    """Маленький проект: репозиторий urec (repo), поддельный форк (fork) и модель (models)."""
    repo = tmp_path / "repo"
    (repo / "envs").mkdir(parents=True)
    for name in pipeline.LOCK_FILES:
        (repo / "envs" / name).write_text(f"пакеты {name}\n", encoding="utf-8")
    (repo / "envs" / "models.lock.json").write_text(json.dumps({BASE_MODEL: "abc123"}))
    git_init(repo)
    shutil.copytree(FAKE_FORK, tmp_path / "fork")
    git_init(tmp_path / "fork")
    model = tmp_path / "models" / "tofu_Llama-3.2-1B-Instruct_full"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    return tmp_path


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    """Все тесты — как на машине без GPU, даже если nvidia-smi есть на этом компьютере."""
    monkeypatch.setattr(pipeline, "GPU_MONITOR", ["no-such-program-nvidia-smi"])
    monkeypatch.setattr(pipeline, "_nvidia_smi", lambda *args: None)


def project_overrides(project):
    """Добавки конфига: пути маленького проекта, а python окружения unl — текущий python.

    Пути — в кавычках: так Hydra принимает в них любые знаки, например «~».
    """
    return [
        f"paths.root='{(project / 'repo').as_posix()}'",
        f"paths.fork_dir='{(project / 'fork').as_posix()}'",
        f"paths.models='{(project / 'models').as_posix()}'",
        f"envs.unl='{Path(sys.executable).as_posix()}'",
    ]


def config(project, *changes):
    """Конфиг urec для маленького проекта; changes — ещё добавки «ключ=значение»."""
    return load_config([*project_overrides(project), *changes])


def raw(project, spec):
    """Папка запуска results/raw/{run} маленького проекта."""
    return project / "repo" / "results" / "raw" / spec.run


def log_text(project, spec, step):
    """Текст лога шага step."""
    return (project / "repo" / "logs" / spec.run / f"{step}.log").read_text(encoding="utf-8")


def manifest_of(project, spec):
    """Манифест запуска spec."""
    return read_json(raw(project, spec) / "manifest.json", Manifest)


def test_run_name_follows_the_plan():
    name = pipeline.run_name("exp2", "3B", "forget05", "NPO", "base", 1)
    assert name == "exp2_3B_f05_NPO_base_s1"  # пример имени из плана


def test_make_spec_fills_name_and_target_model():
    spec = pipeline.make_spec("exp0", "3B", "forget10", "GradDiff")
    assert spec.run == "exp0_3B_f10_GradDiff_base_s0"
    assert spec.base_model == "open-unlearning/tofu_Llama-3.2-3B-Instruct_full"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"split": "forget02"}, "неизвестный сплит forget02"),
        ({"model_size": "7B"}, "неизвестный размер модели 7B"),
        ({"seed": 2}, "должно быть exp0_1B_f01_NPO_base_s2"),  # имя не поменяли вслед за сидом
    ],
    ids=["split", "model-size", "run-name"],
)
def test_check_spec_finds_mistakes(changes, message):
    spec = replace(pipeline.make_spec("exp0", "1B", "forget01", "NPO"), **changes)
    with pytest.raises(ValueError, match=message):
        pipeline.check_spec(spec)


def test_fork_overrides_follow_the_plan(project):
    cfg = config(project)
    spec = pipeline.make_spec("exp1", "3B", "forget05", "NPO", seed=1, unlearn_overrides=["x=1"])
    model = (project / "models" / "tofu_Llama-3.2-3B-Instruct_full").as_posix()
    checkpoint = (project / "fork" / "saves" / "unlearn" / "exp1_3B_f05_NPO_base_s1").as_posix()
    assert pipeline.fork_overrides(cfg, spec) == [
        "model=Llama-3.2-3B-Instruct",
        f"model.model_args.pretrained_model_name_or_path={model}",
        f"model.tokenizer_args.pretrained_model_name_or_path={model}",
        "forget_split=forget05",
        "retain_split=retain95",
        "holdout_split=holdout05",
        "trainer=NPO",
        "retain_logs_path=saves/eval/tofu_Llama-3.2-3B-Instruct_retain95/TOFU_EVAL.json",
        "task_name=exp1_3B_f05_NPO_base_s1",
        "trainer.args.seed=1",
        f"paths.output_dir={checkpoint}",
        *cfg.fork.unlearn_overrides,  # общие добавки из configs/fork/default.yaml
        "x=1",  # добавки запуска — последними
    ]
    assert "model.model_args.torch_dtype=float32" in cfg.fork.unlearn_overrides  # решение части D


def test_run_unlearns_then_evaluates_the_checkpoint(project):
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    pipeline.run(cfg, spec)

    folder = raw(project, spec)
    checkpoint = project / "fork" / "saves" / "unlearn" / spec.run
    assert (checkpoint / "config.json").exists()  # «модель» после забывания
    assert is_done(folder / "unlearn") and is_done(folder / "eval")
    assert (folder / "unlearn" / "fork_config.yaml").exists()  # снимки конфигов форка
    assert (folder / "eval" / "fork_config.yaml").exists()
    assert read_yaml(folder / "spec.yaml", RunSpec) == spec
    assert "allow_dirty: false" in (folder / "config_snapshot.yaml").read_text(encoding="utf-8")

    # оценка получила чекпоинт, а токенизатор — от TOFU-модели, и сплиты из одной тройки
    received = json.loads((folder / "eval" / "TOFU_EVAL.json").read_text())
    model = project / "models" / "tofu_Llama-3.2-1B-Instruct_full"
    assert received["model.model_args.pretrained_model_name_or_path"] == checkpoint.as_posix()
    assert received["model.tokenizer_args.pretrained_model_name_or_path"] == model.as_posix()
    assert (received["forget_split"], received["holdout_split"]) == ("forget01", "holdout01")
    retain_logs = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json"  # от M_ret
    assert received["retain_logs_path"] == retain_logs
    assert received["eval.tofu.overwrite"] == "true"  # оценка считается заново
    assert received["paths.output_dir"] == (folder / "eval").as_posix()

    assert "fake train:" in log_text(project, spec, "unlearn")  # вывод форка — в лог шага
    assert log_text(project, spec, "eval").count("\n=== ") == 1  # заголовок одной попытки


def test_manifest_records_code_environment_and_steps(project):
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    pipeline.run(cfg, spec)
    manifest = manifest_of(project, spec)
    assert len(manifest.git["urec"]) == 40 and len(manifest.git["fork"]) == 40  # полные коммиты
    assert len(manifest.lock_sha) == 64
    assert manifest.model_revisions == {BASE_MODEL: "abc123"}
    assert manifest.hardware.startswith("GPU нет")
    assert list(manifest.steps) == ["unlearn", "eval"]
    for step in manifest.steps.values():
        assert step.status == "done"
        assert step.command[0] == Path(sys.executable).as_posix()  # python окружения unl
        assert step.seconds >= 0 and step.finished >= step.started
        assert step.peak_gpu_mib is None  # GPU нет
    # после python — программа форка, имя конфига Hydra и пресет эксперимента TOFU
    unlearn_start = [
        "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default",
    ]
    assert manifest.steps["unlearn"].command[1:4] == unlearn_start
    eval_start = ["src/eval.py", "--config-name=eval.yaml", "experiment=eval/tofu/default"]
    assert manifest.steps["eval"].command[1:4] == eval_start


def test_manifest_records_peak_gpu_memory(project, monkeypatch):
    monkeypatch.setattr(pipeline, "_stop_gpu_monitor", lambda monitor, path: 3400)  # «замер»
    spec = pipeline.make_spec("exp0", "1B", "forget01", pipeline.TARGET)
    pipeline.run(config(project), spec)
    assert manifest_of(project, spec).steps["eval"].peak_gpu_mib == 3400


def test_target_model_is_only_evaluated(project):
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget05", pipeline.TARGET)
    pipeline.run(cfg, spec)
    folder = raw(project, spec)
    assert not (folder / "unlearn").exists()  # забывания нет
    received = json.loads((folder / "eval" / "TOFU_EVAL.json").read_text())
    model = project / "models" / "tofu_Llama-3.2-1B-Instruct_full"
    assert received["model.model_args.pretrained_model_name_or_path"] == model.as_posix()
    assert received["holdout_split"] == "holdout05"


def test_done_steps_are_skipped(project):
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    pipeline.run(cfg, spec)
    pipeline.run(cfg, spec)  # второй запуск: оба шага уже с DONE
    assert log_text(project, spec, "unlearn").count("\n=== ") == 1  # шаги не запускались снова
    assert log_text(project, spec, "eval").count("\n=== ") == 1


def test_failed_step_stops_the_run_and_the_next_run_continues(project, monkeypatch):
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    folder = raw(project, spec)

    monkeypatch.setenv("FAKE_FORK_FAIL", "eval")  # оценка падает
    with pytest.raises(pipeline.StepFailed) as error:
        pipeline.run(cfg, spec)
    assert "шаг eval завершился с кодом 1" in str(error.value)
    assert "fake eval: ошибка оценки" in str(error.value)  # в сообщении — конец лога
    assert is_done(folder / "unlearn") and not is_done(folder / "eval")
    failed = manifest_of(project, spec).steps["eval"]
    assert failed.status == "failed" and failed.finished is not None

    monkeypatch.delenv("FAKE_FORK_FAIL")  # ошибку исправили — запускаем ту же команду снова
    pipeline.run(cfg, spec)
    assert log_text(project, spec, "unlearn").count("\n=== ") == 1  # обучение не повторялось
    assert log_text(project, spec, "eval").count("\n=== ") == 2  # лог хранит обе попытки
    assert is_done(folder / "eval")
    steps = manifest_of(project, spec).steps
    assert (steps["unlearn"].status, steps["eval"].status) == ("done", "done")  # история цела


def test_step_without_result_is_not_done(project, monkeypatch):
    monkeypatch.setenv("FAKE_FORK_EMPTY", "eval")  # код 0, но результата нет
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget01", pipeline.TARGET)
    with pytest.raises(RuntimeError, match="не создал"):
        pipeline.run(cfg, spec)
    assert not is_done(raw(project, spec) / "eval")
    assert manifest_of(project, spec).steps["eval"].status == "failed"


def test_missing_checkpoint_is_an_error(project, monkeypatch):
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    monkeypatch.setenv("FAKE_FORK_FAIL", "eval")
    with pytest.raises(pipeline.StepFailed):
        pipeline.run(cfg, spec)  # unlearn выполнен, eval упал
    monkeypatch.delenv("FAKE_FORK_FAIL")
    shutil.rmtree(project / "fork" / "saves" / "unlearn" / spec.run)  # чекпоинт пропал
    with pytest.raises(FileNotFoundError, match="удалите"):
        pipeline.run(cfg, spec)


def test_model_must_be_downloaded(project):
    shutil.rmtree(project / "models")
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    with pytest.raises(FileNotFoundError, match="скачайте"):
        pipeline.run(config(project), spec)
    assert not raw(project, spec).exists()  # до запуска шагов ничего не создано


def test_dirty_tree_is_an_error_unless_allowed(project):
    train = project / "fork" / "src" / "train.py"
    train.write_text(train.read_text(encoding="utf-8") + "# правка\n", encoding="utf-8")
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    with pytest.raises(RuntimeError, match="незакоммиченные правки в fork"):
        pipeline.run(config(project), spec)
    assert not raw(project, spec).exists()  # отказ — до записи каких-либо файлов
    pipeline.run(config(project, "allow_dirty=true"), spec)  # разрешили явно
    manifest = manifest_of(project, spec)
    assert manifest.git["fork"].endswith("+dirty")
    assert not manifest.git["urec"].endswith("+dirty")


def test_git_commit_ignores_untracked_files(project):
    fork = project / "fork"
    clean = pipeline.git_commit(fork)
    (fork / "new_log.txt").write_text("новый файл")  # неотслеживаемый файл код не меняет
    assert pipeline.git_commit(fork) == clean


def test_same_name_with_other_spec_is_an_error(project):
    cfg = config(project)
    spec = pipeline.make_spec("exp0", "1B", "forget01", "NPO")
    pipeline.run(cfg, spec)
    other = replace(spec, unlearn_overrides=["trainer.args.learning_rate=5e-5"])  # имя то же
    with pytest.raises(ValueError, match="уже другой spec"):
        pipeline.run(cfg, other)


def test_main_runs_a_spec_file(project, tmp_path, monkeypatch, capsys):
    spec = pipeline.make_spec("exp0", "1B", "forget01", pipeline.TARGET)
    path = tmp_path / "spec.yaml"
    write_yaml(path, spec)
    overrides = project_overrides(project)
    pipeline.main(["run", "--spec", str(path), *overrides])
    assert "все шаги выполнены" in capsys.readouterr().out
    assert read_yaml(raw(project, spec) / "spec.yaml", RunSpec) == spec  # spec — в папке запуска

    other = pipeline.make_spec("exp0", "1B", "forget05", pipeline.TARGET)
    write_yaml(path, other)
    monkeypatch.setenv("FAKE_FORK_FAIL", "eval")
    with pytest.raises(SystemExit, match="шаг eval завершился с кодом 1"):
        pipeline.main(["run", *overrides, "--spec", str(path)])  # --spec можно и после добавок


HEADERS = [  # шапка nvidia-smi у драйверов разных лет
    "| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4     |",
    "| NVIDIA-SMI 610.78       KMD Version: 610.78          CUDA UMD Version: 12.4 |",
]


@pytest.mark.parametrize("header", HEADERS, ids=["old-driver", "new-driver"])
def test_hardware_from_nvidia_smi(monkeypatch, header):
    gpus = "NVIDIA A100-SXM4-40GB, 550.54.15\n"  # ответ на --query-gpu=name,driver_version

    def fake_nvidia_smi(*args):
        return gpus if args else header  # с аргументами — список GPU, без них — шапка

    monkeypatch.setattr(pipeline, "_nvidia_smi", fake_nvidia_smi)
    assert pipeline.hardware() == "NVIDIA A100-SXM4-40GB, драйвер 550.54.15; CUDA 12.4"


def test_gpu_monitor_finds_the_peak(tmp_path, monkeypatch):
    # поддельный nvidia-smi: печатает память двух замеров и «работает» дальше
    fake = "print(1200, flush=True); print(3400, flush=True); import time; time.sleep(60)"
    monkeypatch.setattr(pipeline, "GPU_MONITOR", [sys.executable, "-c", fake])
    path = tmp_path / "gpu.log"
    monitor = pipeline._start_gpu_monitor(path)
    deadline = time.monotonic() + 20
    while "3400" not in path.read_text() and time.monotonic() < deadline:
        time.sleep(0.1)  # ждём, пока программа запустится и напечатает замеры
    assert pipeline._stop_gpu_monitor(monitor, path) == 3400
    assert monitor.poll() is not None  # замер остановлен
