from pathlib import Path

import pytest

from llm_distiller.core.utils import backup_file, compress_content, expand_content


def test_compress_content_gz() -> None:
    data = "test content for compression"
    compressed = compress_content(data, level=9, compression_algo="gz")
    assert compressed != data
    assert isinstance(compressed, bytes)


def test_compress_content_lz4() -> None:
    data = "test content for compression"
    compressed = compress_content(data, level=1, compression_algo="lz4")
    assert compressed != data
    assert isinstance(compressed, bytes)


def test_compress_content_zstd() -> None:
    data = "test content for compression"
    compressed = compress_content(data, level=3, compression_algo="zstd")
    assert compressed != data
    assert isinstance(compressed, bytes)


def test_compress_content_no_compression() -> None:
    data = "test content"
    compressed = compress_content(data, level=0)
    assert compressed == data


def test_expand_content_gz() -> None:
    data = "test content for expansion"
    compressed = compress_content(data, level=9, compression_algo="gz")
    expanded = expand_content(compressed, compression_algo="gz")
    assert expanded == data


def test_expand_content_lz4() -> None:
    data = "test content for expansion"
    compressed = compress_content(data, level=1, compression_algo="lz4")
    expanded = expand_content(compressed, compression_algo="lz4")
    assert expanded == data


def test_expand_content_zstd() -> None:
    data = "test content for expansion"
    compressed = compress_content(data, level=3, compression_algo="zstd")
    expanded = expand_content(compressed, compression_algo="zstd")
    assert expanded == data


def test_expand_content_no_compression() -> None:
    data = "test content"
    expanded = expand_content(data)
    assert expanded == data


def test_backup_file_creates_backup(tmp_path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    result = backup_file(str(test_file), str(backup_dir))

    assert Path(result).exists()
    assert Path(result).read_text() == "test content"


def test_backup_file_nonexistent_raises(tmp_path) -> None:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        backup_file(str(tmp_path / "nonexistent.txt"), str(backup_dir))


def test_backup_file_keeps_max_3_backups(tmp_path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    for _ in range(5):
        backup_file(str(test_file), str(backup_dir))

    backups = list(backup_dir.glob(".test.*.txt.bkp"))
    assert len(backups) <= 3
