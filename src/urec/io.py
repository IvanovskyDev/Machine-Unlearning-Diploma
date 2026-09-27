"""Запись и чтение файлов проекта (план, блоки 22 и 25).

Правила из блока 22:
- файл пишется под временным именем и одним шагом переименовывается в итоговое,
  поэтому оборванный запуск не оставит полузаписанный файл, похожий на готовый;
- маркер DONE ставится последним: папка с DONE — выполненный шаг;
- JSONL — по одной записи JSON на строку; пишется через orjson (буквы не-ASCII
  не экранируются), большие журналы сжимаются zstd;
- файл с одной записью — JSON (manifest.json) или YAML (spec.yaml, его правят руками);
- чтение проверяет schema_version и набор полей.
"""

import hashlib
import io
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import orjson
import yaml
import zstandard

from urec.types import Record

T = TypeVar("T", bound=Record)  # класс записи: объекты какого класса просили, такие и вернутся

DONE = "DONE"  # имя файла-маркера выполненного шага


def atomic_write(path: str | Path, data: bytes) -> None:
    """Записывает data в файл path: либо целиком, либо никак."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # создать папку, если её ещё нет
    tmp = path.with_name(path.name + ".tmp")  # временный файл в той же папке
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)  # один шаг: под именем path либо старый файл, либо новый целиком


def write_jsonl(path: str | Path, records: Iterable[Record]) -> None:
    """Записывает записи в файл .jsonl, по одной на строку. Файл пишется атомарно."""
    atomic_write(path, b"".join(_to_line(record) for record in records))


def read_jsonl(path: str | Path, cls: type[T]) -> list[T]:
    """Читает файл .jsonl и возвращает список объектов класса cls."""
    with open(path, "rb") as f:
        return [_from_line(line, cls, f"{path}:{number}") for number, line in enumerate(f, 1)]


def write_json(path: str | Path, record: Record) -> None:
    """Записывает одну запись в файл JSON с отступами (например, manifest.json). Атомарно."""
    atomic_write(path, orjson.dumps(record, option=orjson.OPT_INDENT_2) + b"\n")


def read_json(path: str | Path, cls: type[T]) -> T:
    """Читает файл JSON с одной записью и возвращает объект класса cls."""
    return _from_line(Path(path).read_bytes(), cls, str(path))


def write_yaml(path: str | Path, record: Record) -> None:
    """Записывает одну запись в файл YAML (например, spec.yaml). Атомарно."""
    data = orjson.loads(orjson.dumps(record))  # запись → словарь: так же, как её видит JSON
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)  # поля — в порядке класса
    atomic_write(path, text.encode("utf-8"))


def read_yaml(path: str | Path, cls: type[T]) -> T:
    """Читает файл YAML с одной записью и возвращает объект класса cls."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _from_data(data, cls, str(path))


class JsonlZstWriter:
    """Потоковая запись больших журналов .jsonl.zst (например, nodes.jsonl.zst).

    Пока запись идёт, данные лежат в файле с окончанием .part, и каждые flush_every
    записей сбрасываются на диск. close() даёт файлу итоговое имя. Удобнее всего через with:

        with JsonlZstWriter(path) as writer:
            writer.write(node)

    Если внутри with случилась ошибка, итоговый файл не появится, а записанное до
    ошибки останется в .part для разбора.
    """

    def __init__(self, path: str | Path, flush_every: int = 100) -> None:
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")  # nodes.jsonl.zst.part
        self.flush_every = flush_every
        self.count = 0  # сколько записей уже записано
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # сжатие «на лету»: данные сжимаются и уходят в файл .part по мере записи
        self.stream = zstandard.ZstdCompressor().stream_writer(open(self.part, "wb"))

    def write(self, record: Record) -> None:
        """Дописывает одну запись."""
        self.stream.write(_to_line(record))
        self.count += 1
        if self.count % self.flush_every == 0:
            self.stream.flush()  # сбросить на диск всё, что уже сжато

    def close(self) -> None:
        """Завершает сжатие, закрывает файл и даёт ему итоговое имя."""
        self.stream.close()  # закрывает и сам файл .part
        os.replace(self.part, self.path)

    def __enter__(self) -> "JsonlZstWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()  # внутри with всё прошло без ошибок
        else:
            self.stream.close()  # ошибка: файл закрыть, но итоговое имя не давать


def read_jsonl_zst(path: str | Path, cls: type[T]) -> Iterator[T]:
    """Читает журнал .jsonl.zst по одной записи, не распаковывая его в память целиком."""
    with open(path, "rb") as f:
        stream = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        text = io.TextIOWrapper(stream, encoding="utf-8")  # распакованные байты → строки текста
        for number, line in enumerate(text, 1):
            yield _from_line(line, cls, f"{path}:{number}")


def mark_done(folder: str | Path) -> None:
    """Ставит в папку маркер DONE: шаг выполнен. Вызывается последним, после всех файлов."""
    atomic_write(Path(folder) / DONE, b"")


def is_done(folder: str | Path) -> bool:
    """Выполнен ли шаг: есть ли в папке маркер DONE."""
    return (Path(folder) / DONE).exists()


def sha256_file(path: str | Path) -> str:
    """Отпечаток файла sha256 (64 шестнадцатеричных знака) — для lock-файлов и данных."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _to_line(record: Record) -> bytes:
    """Запись → одна строка JSON с переводом строки в конце."""
    return orjson.dumps(record) + b"\n"


def _from_line(line: bytes | str, cls: type[T], where: str) -> T:
    """Строка JSON → объект класса cls. where — место в файле для сообщения об ошибке."""
    try:
        data = orjson.loads(line)
    except orjson.JSONDecodeError as error:  # например, строка оборвана на середине
        raise ValueError(f"{where}: строка не читается как JSON: {error}") from error
    return _from_data(data, cls, where)


def _from_data(data: object, cls: type[T], where: str) -> T:
    """Прочитанные из файла данные → объект класса cls; проверяет форму, версию и поля."""
    if not isinstance(data, dict):
        kind = type(data).__name__  # например, list: в файле список, а не запись
        raise ValueError(f"{where}: ожидалась запись {{поле: значение, …}}, а прочитан {kind}")
    if data.get("schema_version") != cls.schema_version:
        raise ValueError(
            f"{where}: schema_version {data.get('schema_version')}, "
            f"а код знает {cls.__name__} версии {cls.schema_version}"
        )
    try:
        return cls(**data)
    except (TypeError, ValueError) as error:  # нет нужного поля, есть лишнее или значения не те
        raise ValueError(f"{where}: запись не подходит к {cls.__name__}: {error}") from error
