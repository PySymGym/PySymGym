"""Unit tests for ``ModelGameManager`` with the process/filesystem seams faked.

No process is spawned and no real output directory is touched: the manager is
built with the shared ``fake_namespace``/``fake_proc`` fixtures and a
``tmp_path`` ``svms_output_path``.
"""

import logging
import socket

import pytest
import torch
from common.classes import GameFailed, GameResult
from common.game import GameMap
from ml.validation.coverage.game_managers.model import process_game_manager as pgm
from ml.validation.coverage.game_managers.model.classes import (
    GameFailedDetails,
    GameResultDetails,
    ModelGameMapInfo,
    SVMConnectionInfo,
)
from ml.validation.coverage.game_managers.model.process_game_manager import (
    ModelGameManager,
)

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def _manager(fake_namespace, tmp_path) -> ModelGameManager:
    return ModelGameManager(
        fake_namespace, torch.nn.Linear(1, 1), svms_output_path=tmp_path
    )


def _write_result(tmp_path, game_map: GameMap, content: str) -> None:
    output_dir = tmp_path / game_map.MapName
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"{game_map.MapName}result").write_text(content)


def test_get_result_parses_and_subtracts_steps_to_start(
    fake_namespace, game_map2svm_factory, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = game_map2svm_factory(steps_to_play=10, steps_to_start=3)
    _write_result(tmp_path, game_map2svm.GameMap, "80 5 7 2")

    result = manager._get_result(game_map2svm, fake_proc())

    assert result == GameResult(
        steps_count=4,
        tests_count=5,
        errors_count=2,
        actual_coverage_percent=80,
    )


def test_get_result_returns_full_coverage_result(
    fake_namespace, game_map2svm_factory, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = game_map2svm_factory()
    _write_result(tmp_path, game_map2svm.GameMap, "100 5 3 0")

    result = manager._get_result(game_map2svm, fake_proc())

    assert isinstance(result, GameResult)
    assert result.actual_coverage_percent == 100
    assert result.steps_count == 3


def test_get_result_immediate_gameover_returns_full_steps(
    fake_namespace, game_map2svm_factory, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = game_map2svm_factory(steps_to_play=10)
    _write_result(tmp_path, game_map2svm.GameMap, "0 0 0 0")

    result = manager._get_result(game_map2svm, fake_proc())

    assert result == GameResult(
        steps_count=10,
        tests_count=0,
        errors_count=0,
        actual_coverage_percent=0,
    )


def test_get_result_malformed_file_returns_game_failed(
    fake_namespace, game_map2svm_factory, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = game_map2svm_factory()
    _write_result(tmp_path, game_map2svm.GameMap, "not numbers")

    result = manager._get_result(game_map2svm, fake_proc(poll_result=0))

    assert isinstance(result, GameFailed)
    assert "Incorrect result" in result.reason


def test_get_result_missing_file_after_exit_returns_game_failed(
    fake_namespace, game_map2svm_factory, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = game_map2svm_factory()

    result = manager._get_result(game_map2svm, fake_proc(poll_result=0))

    assert isinstance(result, GameFailed)
    assert "cannot be found" in result.reason


def test_get_result_retries_while_process_runs(
    fake_namespace,
    game_map2svm_factory,
    tmp_path,
    fake_proc,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = game_map2svm_factory()
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


def test_get_game_steps_unknown_map_returns_none(
    fake_namespace, game_map_factory, tmp_path
) -> None:
    manager = _manager(fake_namespace, tmp_path)

    assert manager.get_game_steps(game_map_factory("Unknown")) is None


def test_get_game_steps_converts_and_caches(
    fake_namespace, game_map_factory, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map = game_map_factory("M")
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None, total_steps=[], proc=None, game_result=None
    )
    monkeypatch.setattr(pgm, "get_steps_from_svm", lambda **kwargs: ["raw"])
    monkeypatch.setattr(pgm, "convert_steps_to_hetero", lambda steps: ["hetero"])

    assert manager.get_game_steps(game_map) == ["hetero"]
    assert manager._games_info["M"].total_steps == ["hetero"]


def test_get_game_steps_returns_none_when_conversion_fails(
    fake_namespace, game_map_factory, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map = game_map_factory("M")
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None, total_steps=[], proc=None, game_result=None
    )

    def boom(**kwargs):
        raise RuntimeError("bad steps")

    monkeypatch.setattr(pgm, "get_steps_from_svm", boom)

    assert manager.get_game_steps(game_map) is None


def test_kill_game_process_kills_and_logs_output(
    fake_namespace, tmp_path, fake_proc
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    proc = fake_proc(output=("out text", "err text"))
    logged: list[str] = []

    manager._kill_game_process(proc, logged.append)

    assert proc.killed is True
    assert logged == ["out:\nout text\nerr:\nerr text"]


def test_delete_game_artifacts_removes_dir_and_info(
    fake_namespace, game_map_factory, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map = game_map_factory("M")
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None, total_steps=[], proc=None, game_result=None
    )
    removed: list = []
    monkeypatch.setattr(pgm, "delete_dir", removed.append)

    manager.delete_game_artifacts(game_map)

    assert removed == [tmp_path / "M"]
    assert "M" not in manager._games_info


def test_run_game_process_formats_and_launches(
    fake_namespace, game_map2svm_factory, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    game_map2svm = game_map2svm_factory(
        map_name="Map1", steps_to_play=3, steps_to_start=1
    )
    game_map2svm.SVMInfo.launch_command = (
        "run --port {Port} --map {MapName} --cover {NameOfObjectToCover}"
    )
    server_socket = object()
    monkeypatch.setattr(
        pgm, "look_for_free_port_locked", lambda lock, svm_info: (4000, server_socket)
    )
    popen_calls: list[tuple] = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return "proc"

    monkeypatch.setattr(pgm.subprocess, "Popen", fake_popen)

    proc, port, returned_socket = manager._run_game_process(game_map2svm)

    assert (proc, port, returned_socket) == ("proc", 4000, server_socket)
    args, kwargs = popen_calls[0]
    assert args == ["run", "--port", "4000", "--map", "Map1", "--cover", "Method"]
    assert kwargs["encoding"] == "utf-8"


class FakeConnectionSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.shutdowns: list[int] = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def shutdown(self, how: int) -> None:
        self.shutdowns.append(how)


class FakeServerSocket:
    def __init__(self, connection: FakeConnectionSocket) -> None:
        self._connection = connection
        self.accepted = 0

    def accept(self):
        self.accepted += 1
        return self._connection, ("localhost", 0)


def _game_result_details(server_socket) -> GameResultDetails:
    return GameResultDetails(
        game_result=GameResult(1, 0, 0, 100),
        svm_connection_info=SVMConnectionInfo(occupied_port=1, socket=server_socket),
    )


def test_notify_steps_requirement_sends_the_flag(
    fake_namespace, game_map_factory, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    connection = FakeConnectionSocket()
    server_socket = FakeServerSocket(connection)
    game_map = game_map_factory("M")
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None,
        total_steps=[],
        proc=None,
        game_result=_game_result_details(server_socket),
    )
    monkeypatch.setattr(manager, "_get_and_log_proc_output", lambda proc, logger: None)

    manager.notify_steps_requirement(game_map, True)
    manager.notify_steps_requirement(game_map, False)

    assert connection.sent == [bytes([1]), bytes([0])]
    assert connection.shutdowns == [socket.SHUT_WR, socket.SHUT_WR]
    assert server_socket.accepted == 2


def test_notify_steps_requirement_unknown_map_is_a_noop(
    fake_namespace, game_map_factory, tmp_path
) -> None:
    manager = _manager(fake_namespace, tmp_path)

    manager.notify_steps_requirement(game_map_factory("Unknown"), True)


def test_notify_steps_requirement_failed_game_logs_output(
    fake_namespace, game_map_factory, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_namespace, tmp_path)
    manager._games_info["M"] = ModelGameMapInfo(
        total_game_state=None,
        total_steps=[],
        proc=None,
        game_result=GameFailedDetails(game_failed=GameFailed("no game")),
    )
    logged: list[tuple] = []
    monkeypatch.setattr(
        manager,
        "_get_and_log_proc_output",
        lambda proc, logger: logged.append((proc, logger)),
    )

    manager.notify_steps_requirement(game_map_factory("M"), True)

    assert logged == [(None, logging.error)]
