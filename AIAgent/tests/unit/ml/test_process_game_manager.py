"""Unit tests for ``ModelGameManager`` with the process/filesystem seams faked.

No process is spawned and no real output directory is touched: the manager is
built with the shared ``fake_namespace``/``fake_proc`` fixtures and a
``tmp_path`` ``svms_output_path``.
"""

import pytest
import torch
from common.classes import GameFailed, GameResult
from common.game import GameMap, GameMap2SVM
from common.validation_coverage.svm_info import SVMInfo
from ml.validation.coverage.game_managers.model import process_game_manager as pgm
from ml.validation.coverage.game_managers.model.classes import ModelGameMapInfo
from ml.validation.coverage.game_managers.model.process_game_manager import (
    ModelGameManager,
)

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def _game_map(
    map_name: str = "MapName", steps_to_play: int = 10, steps_to_start: int = 0
) -> GameMap:
    return GameMap(
        StepsToPlay=steps_to_play,
        StepsToStart=steps_to_start,
        AssemblyFullName="assembly",
        NameOfObjectToCover="Method",
        DefaultSearcher="BFS",
        MapName=map_name,
    )


def _game_map2svm(**kwargs) -> GameMap2SVM:
    return GameMap2SVM(
        GameMap=_game_map(**kwargs),
        SVMInfo=SVMInfo(
            name="svm",
            launch_command="run",
            server_working_dir="/tmp",
            min_port=1,
            max_port=2,
        ),
    )


def _manager(fake_namespace, tmp_path) -> ModelGameManager:
    return ModelGameManager(
        fake_namespace, torch.nn.Linear(1, 1), svms_output_path=tmp_path
    )


def _write_result(tmp_path, game_map: GameMap, content: str, exists: bool = True):
    output_dir = tmp_path / game_map.MapName
    output_dir.mkdir(exist_ok=True)
    if exists:
        (output_dir / f"{game_map.MapName}result").write_text(content)


def test_get_result_parses_and_subtracts_steps_to_start(
    fake_namespace, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = _game_map2svm(steps_to_play=10, steps_to_start=3)
    _write_result(tmp_path, game_map2svm.GameMap, "80 5 7 2")

    result = manager._get_result(game_map2svm, fake_proc())

    assert result == GameResult(
        steps_count=4,
        tests_count=5,
        errors_count=2,
        actual_coverage_percent=80,
    )


def test_get_result_returns_full_coverage_result(
    fake_namespace, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = _game_map2svm()
    _write_result(tmp_path, game_map2svm.GameMap, "100 5 3 0")

    result = manager._get_result(game_map2svm, fake_proc())

    assert isinstance(result, GameResult)
    assert result.actual_coverage_percent == 100
    assert result.steps_count == 3


def test_get_result_immediate_gameover_returns_full_steps(
    fake_namespace, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = _game_map2svm(steps_to_play=10)
    _write_result(tmp_path, game_map2svm.GameMap, "0 0 0 0")

    result = manager._get_result(game_map2svm, fake_proc())

    assert result == GameResult(
        steps_count=10,
        tests_count=0,
        errors_count=0,
        actual_coverage_percent=0,
    )


def test_get_result_malformed_file_returns_game_failed(
    fake_namespace, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = _game_map2svm()
    _write_result(tmp_path, game_map2svm.GameMap, "not numbers")

    result = manager._get_result(game_map2svm, fake_proc(poll_result=0))

    assert isinstance(result, GameFailed)
    assert "Incorrect result" in result.reason


def test_get_result_missing_file_after_exit_returns_game_failed(
    fake_namespace, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = _game_map2svm()

    result = manager._get_result(game_map2svm, fake_proc(poll_result=0))

    assert isinstance(result, GameFailed)
    assert "cannot be found" in result.reason


def test_get_result_retries_while_process_runs(
    fake_namespace, tmp_path, fake_proc, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = _game_map2svm()
    sleeps: list[float] = []

    def write_on_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        _write_result(tmp_path, game_map2svm.GameMap, "50 1 2 0")

    monkeypatch.setattr(pgm.time, "sleep", write_on_sleep)

    result = manager._get_result(game_map2svm, fake_proc(poll_result=None))

    assert isinstance(result, GameResult)
    assert result.actual_coverage_percent == 50
    assert len(sleeps) == 1


def test_log_proc_output_forwards_both_streams(fake_namespace, tmp_path) -> None:
    manager = _manager(fake_namespace, tmp_path)
    logged: list[str] = []

    manager._log_proc_output(("out text", "err text"), logged.append)

    assert logged == ["out:\nout text\nerr:\nerr text"]


def test_get_and_log_proc_output_without_process(fake_namespace, tmp_path) -> None:
    manager = _manager(fake_namespace, tmp_path)
    logged: list[str] = []

    manager._get_and_log_proc_output(None, logged.append)

    assert logged == ["There is no proc?! Can't log proc output."]


def test_get_game_steps_unknown_map_returns_none(fake_namespace, tmp_path) -> None:
    manager = _manager(fake_namespace, tmp_path)

    assert manager.get_game_steps(_game_map("Unknown")) is None


def test_get_game_steps_converts_and_caches(
    fake_namespace, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map = _game_map("M")
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None, total_steps=[], proc=None, game_result=None
    )
    monkeypatch.setattr(pgm, "get_steps_from_svm", lambda **kwargs: ["raw"])
    monkeypatch.setattr(pgm, "convert_steps_to_hetero", lambda steps: ["hetero"])

    assert manager.get_game_steps(game_map) == ["hetero"]
    assert manager._games_info["M"].total_steps == ["hetero"]


def test_get_game_steps_returns_none_when_conversion_fails(
    fake_namespace, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map = _game_map("M")
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None, total_steps=[], proc=None, game_result=None
    )

    def boom(**kwargs):
        raise RuntimeError("bad steps")

    monkeypatch.setattr(pgm, "get_steps_from_svm", boom)

    assert manager.get_game_steps(game_map) is None


def test_delete_game_artifacts_removes_dir_and_info(
    fake_namespace, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map = _game_map("M")
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None, total_steps=[], proc=None, game_result=None
    )
    removed: list = []
    monkeypatch.setattr(pgm, "delete_dir", removed.append)

    manager.delete_game_artifacts(game_map)

    assert removed == [tmp_path / "M"]
    assert "M" not in manager._games_info
