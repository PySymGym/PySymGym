"""Unit tests for ``common.file_system_utils``."""

from pathlib import Path

import pytest

from common.file_system_utils import (
    create_file,
    create_folders_if_necessary,
    delete_dir,
)

pytestmark = pytest.mark.unit


def test_create_folders_creates_missing_and_keeps_existing(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    nested = tmp_path / "a" / "b"

    create_folders_if_necessary([existing, nested])

    assert existing.is_dir()
    assert nested.is_dir()


def test_create_file_creates_and_truncates(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"

    create_file(target)
    assert target.read_text() == ""

    target.write_text("data")
    create_file(target)
    assert target.read_text() == ""


def test_delete_dir_removes_tree(tmp_path: Path) -> None:
    target = tmp_path / "target"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "file.txt").write_text("x")

    delete_dir(target)

    assert not target.exists()


def test_delete_dir_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        delete_dir(tmp_path / "missing")
