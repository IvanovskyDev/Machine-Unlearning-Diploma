"""Тесты urec.io (план, блоки 22 и 25).

Главное: запись → чтение даёт те же объекты; прерванная запись не оставляет
итогового файла; чужая schema_version — ошибка.
tmp_path — новая пустая временная папка, которую pytest даёт каждому тесту.
"""

from dataclasses import replace

import pytest
import zstandard

from urec.io import (
    JsonlZstWriter,
    atomic_write,
    is_done,
    mark_done,
    read_json,
    read_jsonl,
    read_jsonl_zst,
    read_yaml,
    sha256_file,
    write_json,
    write_jsonl,
    write_yaml,
)
from urec.types import QAItem, RunSpec

from sample_records import ALL_RECORDS, QA_ITEM, RUN_SPEC  # образцы записей


def record_name(record):
    """Подпись варианта теста в выводе pytest: имя класса, например [QAItem]."""
    return type(record).__name__


# parametrize — один и тот же тест для каждой записи из ALL_RECORDS, то есть для каждого класса
@pytest.mark.parametrize("record", ALL_RECORDS, ids=record_name)
def test_jsonl_round_trip(tmp_path, record):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [record, record])
    assert read_jsonl(path, type(record)) == [record, record]


@pytest.mark.parametrize("record", ALL_RECORDS, ids=record_name)
def test_jsonl_zst_round_trip(tmp_path, record):
    path = tmp_path / "records.jsonl.zst"
    with JsonlZstWriter(path, flush_every=2) as writer:  # сброс на диск после 2-й и 4-й записи
        for _ in range(5):
            writer.write(record)
    assert list(read_jsonl_zst(path, type(record))) == [record] * 5


@pytest.mark.parametrize("record", ALL_RECORDS, ids=record_name)
def test_json_and_yaml_round_trip(tmp_path, record):
    write_json(tmp_path / "record.json", record)
    assert read_json(tmp_path / "record.json", type(record)) == record
    write_yaml(tmp_path / "record.yaml", record)
    assert read_yaml(tmp_path / "record.yaml", type(record)) == record


def test_yaml_is_easy_to_read(tmp_path):
    path = tmp_path / "spec.yaml"
    write_yaml(path, RUN_SPEC)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("run: exp2_3B_f05_NPO_base_s1\nexp: exp2\n")  # поля — в порядке класса
    assert "unlearn_overrides:\n- trainer.args.learning_rate=2e-5\n" in text  # список — столбиком


def test_yaml_list_instead_of_record_is_an_error(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("- run: exp2_3B_f05_NPO_base_s1\n", encoding="utf-8")  # список, а не запись
    with pytest.raises(ValueError, match="spec.yaml: ожидалась запись"):
        read_yaml(path, RunSpec)


def test_zst_file_is_compressed(tmp_path):
    plain = tmp_path / "items.jsonl"
    packed = tmp_path / "items.jsonl.zst"
    write_jsonl(plain, [QA_ITEM] * 100)
    with JsonlZstWriter(packed) as writer:
        for _ in range(100):
            writer.write(QA_ITEM)
    assert packed.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"  # первые байты любого файла zstd
    # 100 одинаковых строк сжимаются во много раз
    assert packed.stat().st_size < plain.stat().st_size / 10


def test_flush_puts_records_on_disk(tmp_path):
    writer = JsonlZstWriter(tmp_path / "nodes.jsonl.zst", flush_every=3)
    for _ in range(3):
        writer.write(QA_ITEM)  # после 3-й записи — сброс на диск
    part = (tmp_path / "nodes.jsonl.zst.part").read_bytes()
    text = zstandard.ZstdDecompressor().decompressobj().decompress(part)  # распаковать, что есть
    assert text.count(b"\n") == 3  # все три записи уже на диске, хотя файл ещё не закрыт
    writer.close()


def test_interrupted_write_leaves_no_final_file(tmp_path):
    path = tmp_path / "nodes.jsonl.zst"
    with pytest.raises(RuntimeError):
        with JsonlZstWriter(path) as writer:
            writer.write(QA_ITEM)
            raise RuntimeError("обрыв посреди записи")
    assert not path.exists()  # итогового файла нет
    assert (tmp_path / "nodes.jsonl.zst.part").exists()  # записанное осталось в .part


def test_foreign_schema_version_is_an_error(tmp_path):
    path = tmp_path / "items.jsonl"
    write_jsonl(path, [replace(QA_ITEM, schema_version=2)])  # как будто файл от другой версии кода
    with pytest.raises(ValueError, match="schema_version 2"):
        read_jsonl(path, QAItem)


BAD_LINES = [
    '{"item_id": "forget01/0002", "schema_version": 1}',  # не хватает полей
    '{"item_id": "forget01/0002", "sche',  # строка оборвана — это не JSON
    '["forget01/0002"]',  # список вместо объекта {…}
]


@pytest.mark.parametrize("line", BAD_LINES, ids=["missing-fields", "cut-off", "not-object"])
def test_bad_line_is_an_error(tmp_path, line):
    path = tmp_path / "items.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="items.jsonl:1"):  # в ошибке — файл и номер строки
        read_jsonl(path, QAItem)


def test_non_ascii_is_written_as_is(tmp_path):
    path = tmp_path / "items.jsonl"
    write_jsonl(path, [replace(QA_ITEM, author="Жюль Верн")])
    assert "Жюль Верн".encode() in path.read_bytes()  # буквы как есть, а не \u0416\u044e…
    assert read_jsonl(path, QAItem)[0].author == "Жюль Верн"


def test_atomic_write_replaces_whole_file(tmp_path):
    path = tmp_path / "sub" / "file.txt"  # папки sub ещё нет: atomic_write создаст её
    atomic_write(path, b"old")
    atomic_write(path, b"new")
    assert path.read_bytes() == b"new"
    assert [p.name for p in path.parent.iterdir()] == ["file.txt"]  # временного .tmp не осталось


def test_done_marker(tmp_path):
    assert not is_done(tmp_path)
    mark_done(tmp_path)
    assert is_done(tmp_path)


def test_sha256_file(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    # известный отпечаток sha256 строки "abc" из стандарта SHA-2
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
