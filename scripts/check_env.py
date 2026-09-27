"""Самопроверка окружения unl или atk (план, блок 9): по каждой проверке печатает OK или FAIL.

Скрипт запускает python того окружения, которое проверяется, — только так видны его пакеты:
    /content/envs/unl/bin/python scripts/check_env.py unl
    /content/envs/atk/bin/python scripts/check_env.py atk
Из блокнота его вызывает colab_setup.check_envs(). Если хоть одна проверка не прошла, скрипт
завершается с кодом 1 — по нему check_envs() узнаёт, что что-то не так.

Пакеты окружений импортируются внутри проверок, а не в начале файла: в unl нет vLLM,
а в atk нет FlashAttention, поэтому общий импорт упал бы в любом из них.
"""

import math
import os
import subprocess
import sys

# папка репозитория — на уровень выше папки scripts, где лежит этот файл
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def packages_match_lock(env_name):
    # uv pip freeze — точные версии всех пакетов окружения (sys.executable — его python).
    # Правило то же, что при записи lock-файла (colab_setup.save_lock): без пакетов «из папки»
    # (--exclude-editable, это urec), без FlashAttention и самого OpenUnlearning
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--exclude-editable", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    installed = []
    for line in freeze.splitlines():
        if "flash-attn" not in line and "open-unlearning" not in line:
            installed.append(line)
    with open(f"{REPO}/envs/requirements-{env_name}.lock", encoding="utf-8") as f:
        locked = f.read().splitlines()
    # set — множество строк; разность множеств показывает, что лишнее и чего не хватает
    extra = sorted(set(installed) - set(locked))
    missing = sorted(set(locked) - set(installed))
    assert not extra and not missing, f"лишние: {extra}, не хватает: {missing}"
    return f"{len(installed)} пакетов"


def unl_packages():
    return packages_match_lock("unl")


def atk_packages():
    return packages_match_lock("atk")


def torch_and_flash_attn():
    import flash_attn
    import torch

    assert torch.cuda.is_available(), "torch не видит GPU"
    gpu = torch.cuda.get_device_name(0)
    return f"torch {torch.__version__}, flash-attn {flash_attn.__version__}, {gpu}"


def torch_and_vllm():
    import torch
    import vllm

    assert torch.cuda.is_available(), "torch не видит GPU"
    return f"torch {torch.__version__}, vLLM {vllm.__version__}"


def spacy_finds_person():
    import spacy

    assert spacy.prefer_gpu(), "spaCy не видит GPU"
    nlp = spacy.load("en_core_web_trf")
    doc = nlp("Basil Mahfouz Al-Kuwaiti was born in Kuwait City in 1956.")
    labels = [entity.label_ for entity in doc.ents]
    found = [entity.text + " — " + entity.label_ for entity in doc.ents]
    assert "PERSON" in labels, found
    return "; ".join(found)


def bertscore_f1():
    from bert_score import BERTScorer

    # BERTScore сравнивает два предложения по смыслу; F1 — одно число, чем больше, тем ближе смысл
    scorer = BERTScorer(lang="en", model_type="roberta-large", rescale_with_baseline=True)
    precision, recall, f1 = scorer.score(
        ["He was born in Kuwait."], ["The author was born in Kuwait City."]
    )
    assert math.isfinite(f1.item()), f1
    return f"F1 = {f1.item():.3f}"


def urec_installed():
    import urec

    return "из папки " + urec.__path__[0]


def urec_tests():
    # -m "not gpu and not network" — только тесты, которым не нужны GPU и интернет;
    # -p no:cacheprovider — не оставлять в репозитории папку .pytest_cache
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + ["-m", "not gpu and not network", "tests"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    last_lines = result.stdout.strip().splitlines()[-5:]  # по последним строкам видна причина
    assert result.returncode == 0, "\n".join(last_lines)
    return last_lines[-1]  # итог pytest, например «120 passed in 35.2s»


# какие проверки делать в каждом окружении: (что проверяем, функция проверки)
CHECKS = {
    "unl": [
        ("пакеты совпадают с envs/requirements-unl.lock", unl_packages),
        ("torch видит GPU, FlashAttention импортируется", torch_and_flash_attn),
    ],
    "atk": [
        ("пакеты совпадают с envs/requirements-atk.lock", atk_packages),
        ("torch видит GPU, vLLM импортируется", torch_and_vllm),
        ("spaCy на GPU находит имя человека", spacy_finds_person),
        ("BERTScore считает F1", bertscore_f1),
        ("пакет urec установлен", urec_installed),
        ("тесты urec без GPU и интернета проходят", urec_tests),
    ],
}


def main():
    env_name = sys.argv[1]  # какое окружение проверяем: unl или atk
    fails = 0
    for title, check in CHECKS[env_name]:
        # любая ошибка внутри проверки — это FAIL этой проверки, а не падение всего скрипта
        try:
            details = check()
            print(f"OK    {title}: {details}")
        except Exception as error:
            print(f"FAIL  {title}")
            print(f"      {error}")
            fails += 1
    if fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
