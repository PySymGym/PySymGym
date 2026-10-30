"""Unit tests for the atomic dataset-file replacement helper."""

import os
from pathlib import Path

import pytest
from atomic_io import atomic_write_text

pytestmark = pytest.mark.unit


def test_atomic_write_text_writes_content(tmp_path: Path) -> None:
    target = tmp_path / "dataset.json"

    atomic_write_text(target, '{"a": 1}')

    assert target.read_text(encoding="utf-8") == '{"a": 1}'


def test_atomic_write_text_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "dataset.json"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_temp_lives_in_the_target_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "dataset.json"
    target.write_text("old", encoding="utf-8")
    seen: dict[str, str] = {}
    real_replace = os.replace

    def spy(src: str, dst: str) -> None:
        seen["src"] = src
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)

    atomic_write_text(target, "new")

    assert Path(seen["src"]).parent == target.parent


def test_atomic_write_write_failure_keeps_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "dataset.json"
    target.write_text("original", encoding="utf-8")

    def boom(fd: int) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError, match="write failed"):
        atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_replace_failure_keeps_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "dataset.json"
    target.write_text("original", encoding="utf-8")

    def deny(src: str, dst: str) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", deny)

    with pytest.raises(OSError, match="rename failed"):
        atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_replace_failure_after_temp_vanished_does_not_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "dataset.json"
    target.write_text("original", encoding="utf-8")

    def vanish(src: str, dst: str) -> None:
        os.unlink(src)  # a concurrent process cleaned the temp up
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", vanish)

    with pytest.raises(OSError, match="rename failed"):
        atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_keyboard_interrupt_keeps_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "dataset.json"
    target.write_text("original", encoding="utf-8")

    def interrupt(src: str, dst: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)

    with pytest.raises(KeyboardInterrupt):
        atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]
