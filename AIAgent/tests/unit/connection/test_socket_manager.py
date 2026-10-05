"""Unit tests for the broker socket manager, with psutil/websocket patched.

The websocket is the shared ``fake_websocket`` fake; ``psutil.pid_exists`` is
patched so no real process is inspected. The tests assert the retry loop and,
crucially, that ``return_instance`` runs on every exit path.
"""

import pytest
from common.validation_coverage.svm_info import SVMInfo
from config import GameServerConnectorConfig
from connection.broker_conn import socket_manager
from connection.broker_conn.classes import ServerInstanceInfo
from connection.errors_connection import ProcessStoppedError

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def _svm_info() -> SVMInfo:
    return SVMInfo(
        name="svm",
        launch_command="run",
        server_working_dir="/tmp",
        min_port=1,
        max_port=2,
    )


def _server_instance() -> ServerInstanceInfo:
    return ServerInstanceInfo(svm_name="svm", port=4000, ws_url="ws://host", pid=7)


def test_process_running_yields_for_live_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket_manager.psutil, "pid_exists", lambda pid: True)

    with socket_manager.process_running(123):
        pass


def test_process_running_raises_for_dead_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket_manager.psutil, "pid_exists", lambda pid: False)

    with pytest.raises(ProcessStoppedError):
        with socket_manager.process_running(123):
            pass


def test_wait_for_connection_returns_the_connected_socket(
    monkeypatch: pytest.MonkeyPatch, fake_websocket
) -> None:
    ws = fake_websocket
    monkeypatch.setattr(socket_manager.websocket, "WebSocket", lambda: ws)
    monkeypatch.setattr(socket_manager.psutil, "pid_exists", lambda pid: True)

    instance = _server_instance()
    result = socket_manager.wait_for_connection(instance)

    assert result is ws
    assert ws.url == instance.ws_url
    assert ws.timeout == GameServerConnectorConfig.CREATE_CONNECTION_TIMEOUT_SEC


def test_wait_for_connection_raises_after_retries(
    monkeypatch: pytest.MonkeyPatch, fake_websocket
) -> None:
    ws = fake_websocket
    ws.will_connect = False
    monkeypatch.setattr(socket_manager.websocket, "WebSocket", lambda: ws)
    monkeypatch.setattr(socket_manager.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(socket_manager.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        GameServerConnectorConfig, "WAIT_FOR_SOCKET_RECONNECTION_MAX_RETRIES", 2
    )

    with pytest.raises(RuntimeError, match="Retries exhausted"):
        socket_manager.wait_for_connection(_server_instance())


def test_wait_for_connection_propagates_process_stopped(
    monkeypatch: pytest.MonkeyPatch, fake_websocket
) -> None:
    ws = fake_websocket
    ws.connect_error = ConnectionRefusedError()
    monkeypatch.setattr(socket_manager.websocket, "WebSocket", lambda: ws)
    monkeypatch.setattr(socket_manager.psutil, "pid_exists", lambda pid: False)

    with pytest.raises(ProcessStoppedError):
        socket_manager.wait_for_connection(_server_instance())


def test_game_server_socket_manager_returns_instance_on_success(
    monkeypatch: pytest.MonkeyPatch, fake_websocket
) -> None:
    ws = fake_websocket
    instance = _server_instance()
    returned: list[ServerInstanceInfo] = []
    monkeypatch.setattr(socket_manager, "acquire_instance", lambda svm_info: instance)
    monkeypatch.setattr(
        socket_manager, "wait_for_connection", lambda server_instance: ws
    )
    monkeypatch.setattr(socket_manager, "return_instance", returned.append)

    with socket_manager.game_server_socket_manager(_svm_info()) as sock:
        assert sock is ws

    assert ws.timeout == GameServerConnectorConfig.RESPONCE_TIMEOUT_SEC
    assert ws.closed is True
    assert returned == [instance]


def test_game_server_socket_manager_returns_instance_on_body_failure(
    monkeypatch: pytest.MonkeyPatch, fake_websocket
) -> None:
    ws = fake_websocket
    instance = _server_instance()
    returned: list[ServerInstanceInfo] = []
    monkeypatch.setattr(socket_manager, "acquire_instance", lambda svm_info: instance)
    monkeypatch.setattr(
        socket_manager, "wait_for_connection", lambda server_instance: ws
    )
    monkeypatch.setattr(socket_manager, "return_instance", returned.append)

    with pytest.raises(ValueError):
        with socket_manager.game_server_socket_manager(_svm_info()):
            raise ValueError("body failed")

    assert ws.closed is True
    assert returned == [instance]


def test_game_server_socket_manager_returns_instance_when_connect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _server_instance()
    returned: list[ServerInstanceInfo] = []
    monkeypatch.setattr(socket_manager, "acquire_instance", lambda svm_info: instance)

    def no_connection(server_instance: ServerInstanceInfo):
        raise RuntimeError("no connection")

    monkeypatch.setattr(socket_manager, "wait_for_connection", no_connection)
    monkeypatch.setattr(socket_manager, "return_instance", returned.append)

    with pytest.raises(RuntimeError, match="no connection"):
        with socket_manager.game_server_socket_manager(_svm_info()):
            pass

    assert returned == [instance]
