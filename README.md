# Machine-Unlearning-Diploma

**Evaluating and Improving the Robustness of Machine Unlearning in Large Language Models against Adaptive Multi-Turn Knowledge Recovery Attacks**

Код дипломной работы. Языковую модель заставляют «забыть» часть знаний (unlearning на датасете TOFU), а затем адаптивные многоходовые атаки пытаются эти знания вернуть. Цель — измерить, насколько забывание устойчиво, и сделать его устойчивее. Работа идёт по «Плану кода» (части A–G), вычисления — в Google Colab.

## Блокноты

Каждая часть плана — отдельный блокнот. Открыть его можно кнопкой «Open in Colab» или в Colab через File → Open notebook → GitHub.

| Часть плана | Блокнот | Статус |
|---|---|---|
| A–B. Подготовка: доступы, машина, окружения Python `unl` и `atk` | [`notebooks/A_setup.ipynb`](notebooks/A_setup.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IvanovskyDev/Machine-Unlearning-Diploma/blob/main/notebooks/A_setup.ipynb) | части A и B объединены, команды — в `scripts/colab_setup.py`; нужна проверка в Colab |
| C. Модели и данные TOFU | [`notebooks/C_models_data.ipynb`](notebooks/C_models_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IvanovskyDev/Machine-Unlearning-Diploma/blob/main/notebooks/C_models_data.ipynb) | работает |
| D. Первые запуски руками | [`notebooks/D_first_runs.ipynb`](notebooks/D_first_runs.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IvanovskyDev/Machine-Unlearning-Diploma/blob/main/notebooks/D_first_runs.ipynb) | работает; обучение в шагах 12–13 — на A100 с весами во float32 |
| E. Наш репозиторий, форк OpenUnlearning, архитектура | [`notebooks/E_repository.ipynb`](notebooks/E_repository.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IvanovskyDev/Machine-Unlearning-Diploma/blob/main/notebooks/E_repository.ipynb) | работает |
| F. Пакет `urec`, вехи M0–M8 | веха M0: [`notebooks/F_M0.ipynb`](notebooks/F_M0.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IvanovskyDev/Machine-Unlearning-Diploma/blob/main/notebooks/F_M0.ipynb) | веха M0 выполнена; идёт веха M1: готовы куски 1–3 из 4 — промпт victim, серверы и клиенты vLLM, оркестратор |

Что и зачем делает каждая строка, объяснено для новичка: части A и B — в [`docs/A_B_explained.md`](docs/A_B_explained.md), часть C — в [`docs/C_explained.md`](docs/C_explained.md), часть D — в [`docs/D_explained.md`](docs/D_explained.md), часть E — в [`docs/E_explained.md`](docs/E_explained.md), часть F по вехам — в [`docs/F_M0_explained.md`](docs/F_M0_explained.md) и [`docs/F_M1_explained.md`](docs/F_M1_explained.md).

## Как устроен проект

Код пишется на Windows и отправляется в GitHub, всё, что требует GPU, считается в Google Colab, а результаты сохраняются на Google Drive. Машина Colab временная, поэтому у каждого файла заранее определено место:

| Где | Что лежит | Почему там |
|---|---|---|
| GitHub (этот репозиторий) | код, блокноты, скрипты, конфиги, lock-файлы, итоговые таблицы и рисунки | маленькое, нужна история изменений |
| GitHub, форк [`IvanovskyDev/open-unlearning`](https://github.com/IvanovskyDev/open-unlearning), ветка `tau` | изменения внутри OpenUnlearning; подключён сюда как submodule `external/open-unlearning` | свои изменения фреймворка с историей, версия не меняется без нас |
| Google Drive, `MyDrive/unlearning_data/` | чекпоинты, результаты атак, логи, лабораторный журнал, таблицы для чтения (`data/`), копии lock-файлов для блокнотов C–D (`envs/`; главные — в `envs/` репозитория) | уникальное, не должно пропасть при отключении Colab |
| Диск машины Colab, `/content/` | кэш Hugging Face, скачанные модели, окружения Python | большое, но за минуты скачивается заново |

Правила:

- Код и блокноты правятся в одном месте — на Windows — и уходят в GitHub; в Colab блокноты только открывают и запускают. Так правки не конфликтуют.
- Блокнот выполняется сверху вниз. Машина Colab каждый раз новая, поэтому каждая сессия начинается с ячейки, которая клонирует репозиторий и подключает `scripts/colab_setup.py`, и с `colab_setup.start_session()` (Drive, папки, переменные окружения, токен Hugging Face); окружения `unl` и `atk` заново собирает из lock-файлов `colab_setup.build_envs()`. `pip install -U` в них не делается. Блокноты C–F переводятся на это по очереди.
- Репозиторий клонируется вместе с форком: `git clone --recurse-submodules https://github.com/IvanovskyDev/Machine-Unlearning-Diploma.git`. Правка в форке — это два коммита: в форк и в этот репозиторий (разбор E, раздел 3).
- Каждая часть и веха — в своей ветке и вливается через Pull Request; перед коммитом работают автопроверки pre-commit (разбор E, раздел 5). На GitHub те же проверки и все тесты запускает CI, и PR вливается только с зелёной галочкой (разбор F_M0, раздел 14).
- Токен Hugging Face хранится только в Colab Secrets (`HF_TOKEN`) и никогда не попадает в код.
- Каждый запуск записывается в журнал: где, чем, с какой командой, что получилось.
- Всё новое сначала проверяется на Llama-3.2-1B и forget01, потом на 3B.
- Логика — только в пакете `src/urec` (с вехи M0) и в форке, параметры — в `configs/`, команды запуска — в `scripts/`; блокноты только запускают шаги и рисуют.

## Структура

```
Machine-Unlearning-Diploma/
├── README.md
├── .gitignore                 # веса, чекпоинты, логи и секреты в git не попадают
├── .gitattributes             # скрипты .sh — всегда с переводами строк Linux
├── .gitmodules                # где лежит форк OpenUnlearning и за какой веткой он следит
├── .pre-commit-config.yaml    # автопроверки при коммите
├── .github/workflows/ci.yml   # проверка на GitHub (CI) при каждом PR
├── pyproject.toml             # пакет urec: зависимости, настройки ruff, pytest, mypy
├── docs/                      # разборы частей для новичка: A_B, C, D, E, F_M0, F_M1
├── notebooks/                 # блокноты Colab частей A–F
├── scripts/                   # colab_setup.py (настройка Colab для блокнотов), check_env.py, check_env.sh, measure.sh, start_vllm.sh, stop_vllm.sh
├── envs/                      # lock-файлы окружений, ревизии моделей, пакеты для тестов на CPU
├── external/open-unlearning/  # форк OpenUnlearning, ветка tau (submodule)
├── configs/                   # Hydra-конфиги urec: main, paths, data, envs, fork
├── src/urec/                  # пакет urec: types, io, config, data, cli, serve, pipeline; дальше — по вехам M2–M8
├── tests/                     # тесты на CPU: unit/, integration/ (нужен интернет), golden/, fixtures/
├── data/                      # маленькие данные: items (вопросы TOFU), позже calibration, annotation, retain_regimes
├── results/
│   ├── raw/                   # ссылка на Drive, создаётся в Colab, в git не попадает
│   ├── tables/                # итоговые таблицы
│   └── figures/               # итоговые рисунки
├── analysis/                  # статистика вне Python
└── thesis/                    # исходники диплома в LaTeX
```

Пустые пока папки содержат файл `.gitkeep`. `Makefile` появится позже, в части F.
