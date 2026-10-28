"""Entrypoint seam tests for ``launch_servers.py``.

The broker is driven with fakes: a scripted socket for ``wait_for_port``,
fake requests for the aiohttp handlers, a fake ``Popen`` for the server
lifecycle, and fake ``os.kill``/``psutil`` for the kill paths. No real
processes, ports, or event-loop-bound state are touched.
"""

import asyncio
import json
import logging
import signal
import socket
import subprocess
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace

import psutil
import pytest

import launch_servers as launch_servers_module
from common.validation_coverage.svm_info import SVMInfo
from config import FeatureConfig, OnGameServerRestartFeature
from connection.broker_conn.classes import ServerInstanceInfo, Undefined

pytestmark = pytest.mark.unit


def svm_info(server_working_dir: str = "/srv") -> SVMInfo:
    return SVMInfo(
        name="VSharp",
        launch_command="run --port {port}",
        server_working_dir=server_working_dir,
        min_port=35000,
        max_port=35050,
    )


def svm_query() -> dict:
    """The query dict a client sends to /get_ws (strings, as aiohttp delivers)."""
    return {
        "name": "VSharp",
        "launch_command": "run --port {port}",
        "server_working_dir": "/srv",
        "min_port": "35000",
        "max_port": "35050",
    }


def make_read_request(payload: bytes) -> SimpleNamespace:
    """A fake aiohttp request whose ``read()`` returns ``payload``."""

    async def read() -> bytes:
        return payload

    return SimpleNamespace(read=read)


class FakeSocket:
    """A ``socket.socket`` stand-in with a scripted ``connect_ex``."""

    def __init__(self, connect_ex_result: int) -> None:
        self.connect_ex_result = connect_ex_result
        self.connect_ex_calls: list[tuple] = []

    def connect_ex(self, address: tuple) -> int:
        self.connect_ex_calls.append(address)
        return self.connect_ex_result


def fake_socket_module(monkeypatch: pytest.MonkeyPatch, connect_ex_result: int):
    """Replace ``launch_servers.socket`` with a fake; return the recorder.

    Only the module reference is replaced (not the real ``socket.socket``),
    so asyncio's own socket creation inside ``asyncio.run`` is untouched.
    """
    socks: list[FakeSocket] = []

    def factory(*args, **kwargs) -> FakeSocket:
        sock = FakeSocket(connect_ex_result)
        socks.append(sock)
        return sock

    monkeypatch.setattr(
        launch_servers_module,
        "socket",
        SimpleNamespace(
            socket=factory,
            AF_INET=socket.AF_INET,
            SOCK_STREAM=socket.SOCK_STREAM,
        ),
    )
    return socks


class TestWaitForPort:
    def test_open_port_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socks = fake_socket_module(monkeypatch, 0)

        assert asyncio.run(launch_servers_module.wait_for_port(35012)) is True
        assert socks[0].connect_ex_calls == [("", 35012)]

    def test_closed_port_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socks = fake_socket_module(monkeypatch, 111)

        assert (
            asyncio.run(launch_servers_module.wait_for_port(35012, timeout=0.25))
            is False
        )
        # Polled until the timeout expired.
        assert len(socks[0].connect_ex_calls) >= 2


class TestGetSocketUrl:
    def test_url_shape(self) -> None:
        assert launch_servers_module.get_socket_url(35012) == (
            "ws://0.0.0.0:35012/gameServer"
        )


class TestDequeueInstance:
    def _setup(self, monkeypatch: pytest.MonkeyPatch, queue: Queue) -> None:
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(
            launch_servers_module, "SERVER_INSTANCES", queue, raising=False
        )
        monkeypatch.setattr(
            launch_servers_module.psutil,
            "Process",
            lambda pid: SimpleNamespace(pid=pid),
        )

    def test_success_issues_a_real_instance(self, monkeypatch: pytest.MonkeyPatch):
        queue = Queue()
        queue.put(ServerInstanceInfo("VSharp", 0, "None", pid=Undefined))
        self._setup(monkeypatch, queue)
        issued = ServerInstanceInfo(
            "VSharp", 35012, "ws://0.0.0.0:35012/gameServer", 1234
        )
        calls: list[tuple] = []

        async def fake_run_server_instance(svm_info_arg, should_start_server):
            calls.append((svm_info_arg, should_start_server))
            return issued

        monkeypatch.setattr(
            launch_servers_module, "run_server_instance", fake_run_server_instance
        )

        response = asyncio.run(
            launch_servers_module.dequeue_instance(SimpleNamespace(query=svm_query()))
        )

        # The query is parsed into an SVMInfo (pydantic coerces the strings),
        # and the restart feature (enabled by default) starts the server.
        assert calls == [(svm_info(), True)]
        # The response body is the JSON encoding of the issued info's to_json.
        assert json.loads(response.text) == issued.to_json()
        assert queue.empty()

    def test_empty_queue_raises(self, monkeypatch: pytest.MonkeyPatch):
        self._setup(monkeypatch, Queue())

        with pytest.raises(Empty):
            asyncio.run(
                launch_servers_module.dequeue_instance(
                    SimpleNamespace(query=svm_query())
                )
            )

    def test_failure_requeues_the_placeholder(self, monkeypatch: pytest.MonkeyPatch):
        queue = Queue()
        placeholder = ServerInstanceInfo("VSharp", 0, "None", pid=Undefined)
        queue.put(placeholder)
        self._setup(monkeypatch, queue)

        async def fake_run_server_instance(svm_info_arg, should_start_server):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            launch_servers_module, "run_server_instance", fake_run_server_instance
        )

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(
                launch_servers_module.dequeue_instance(
                    SimpleNamespace(query=svm_query())
                )
            )
        assert queue.get_nowait() is placeholder


class TestEnqueueInstance:
    def test_restart_enabled_kills_and_rewraps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        queue = Queue()
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(
            launch_servers_module, "SERVER_INSTANCES", queue, raising=False
        )
        kill_calls: list[ServerInstanceInfo] = []
        monkeypatch.setattr(
            launch_servers_module, "kill_server", lambda info: kill_calls.append(info)
        )
        original = ServerInstanceInfo(
            "VSharp", 35012, "ws://0.0.0.0:35012/gameServer", 1234
        )

        response = asyncio.run(
            launch_servers_module.enqueue_instance(
                make_read_request(original.to_json().encode("utf-8"))
            )
        )

        assert kill_calls == [original]
        assert queue.get_nowait() == ServerInstanceInfo(
            "VSharp", 35012, "ws://0.0.0.0:35012/gameServer", pid=Undefined
        )
        assert response.status == 200

    def test_restart_disabled_keeps_the_pid(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            FeatureConfig,
            "ON_GAME_SERVER_RESTART",
            OnGameServerRestartFeature(
                enabled=False, wait_for_reset_retries=1, wait_for_reset_time=0.01
            ),
        )
        queue = Queue()
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(
            launch_servers_module, "SERVER_INSTANCES", queue, raising=False
        )
        kill_calls: list[ServerInstanceInfo] = []
        monkeypatch.setattr(
            launch_servers_module, "kill_server", lambda info: kill_calls.append(info)
        )
        original = ServerInstanceInfo(
            "VSharp", 35012, "ws://0.0.0.0:35012/gameServer", 1234
        )

        response = asyncio.run(
            launch_servers_module.enqueue_instance(
                make_read_request(original.to_json().encode("utf-8"))
            )
        )

        assert kill_calls == []
        assert queue.get_nowait() == original
        assert response.status == 200


class TestAppendResults:
    def test_appends_the_decoded_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "RESULTS", [], raising=False)

        response = asyncio.run(
            launch_servers_module.append_results(make_read_request(b'{"r": 1}'))
        )

        assert launch_servers_module.RESULTS == ['{"r": 1}']
        assert response.status == 200


class TestRecvRes:
    def test_empty_results_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "RESULTS", [], raising=False)

        with pytest.raises(RuntimeError, match="Must play a game first"):
            asyncio.run(launch_servers_module.send_and_clear_results(SimpleNamespace()))

    def test_returns_and_clears(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "RESULTS", ["a", "b"], raising=False)

        response = asyncio.run(
            launch_servers_module.send_and_clear_results(SimpleNamespace())
        )

        assert response.text == json.dumps(["a", "b"])
        assert launch_servers_module.RESULTS == []


class TestRunServerInstance:
    def test_should_not_start_returns_placeholder(self) -> None:
        result = asyncio.run(
            launch_servers_module.run_server_instance(
                svm_info(), should_start_server=False
            )
        )

        assert result == ServerInstanceInfo("VSharp", 0, "None", pid=Undefined)

    def test_starts_the_server(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A fresh lock: the module-level one binds to the first event loop.
        monkeypatch.setattr(
            launch_servers_module, "avoid_same_free_port_lock", asyncio.Lock()
        )
        next_free_port_calls: list[tuple] = []

        def fake_next_free_port(min_port, max_port):
            next_free_port_calls.append((min_port, max_port))
            return 35012

        monkeypatch.setattr(
            launch_servers_module, "next_free_port", fake_next_free_port
        )
        popen_calls: list[tuple] = []

        class FakePopen:
            # The nested start_server annotates subprocess.Popen[bytes]; the
            # annotation is evaluated at def time, so the fake must subscript.
            def __class_getitem__(cls, item):
                return cls

            def __init__(self, args, **kwargs):
                popen_calls.append((args, kwargs))
                self.pid = 4321

        monkeypatch.setattr(launch_servers_module.subprocess, "Popen", FakePopen)

        async def fake_wait_for_port(port, timeout=10):
            return True

        monkeypatch.setattr(launch_servers_module, "wait_for_port", fake_wait_for_port)
        procs: list[int] = []
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "PROCS", procs, raising=False)

        result = asyncio.run(
            launch_servers_module.run_server_instance(
                svm_info(server_working_dir=str(tmp_path)), should_start_server=True
            )
        )

        assert next_free_port_calls == [(35000, 35050)]
        (args, kwargs) = popen_calls[0]
        assert args == ["run", "--port", "35012"]
        assert kwargs == {
            "stdout": subprocess.PIPE,
            "start_new_session": True,
            "cwd": Path(tmp_path).absolute(),
        }
        assert procs == [4321]
        assert result == ServerInstanceInfo(
            "VSharp", 35012, "ws://0.0.0.0:35012/gameServer", 4321
        )


class TestRunServers:
    def test_placeholders(self) -> None:
        result = asyncio.run(launch_servers_module.run_servers(3))

        assert result == [
            ServerInstanceInfo("_", 0, "None", pid=Undefined) for _ in range(3)
        ]


class TestKillServer:
    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        procs: list,
        *,
        status=None,
        kill_error=None,
        process_error=None,
        retries=2,
        wait_time=0.01,
    ):
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "PROCS", procs, raising=False)
        monkeypatch.setattr(
            FeatureConfig,
            "ON_GAME_SERVER_RESTART",
            OnGameServerRestartFeature(
                enabled=True,
                wait_for_reset_retries=retries,
                wait_for_reset_time=wait_time,
            ),
        )
        kill_calls: list[tuple] = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            if kill_error is not None:
                raise kill_error

        monkeypatch.setattr(launch_servers_module.os, "kill", fake_kill)
        process_calls: list[int] = []

        class FakeProc:
            def __init__(self, pid):
                self.pid = pid

            def status(self):
                return status

        def fake_process(pid):
            process_calls.append(pid)
            if process_error is not None:
                raise process_error
            return FakeProc(pid)

        monkeypatch.setattr(launch_servers_module.psutil, "Process", fake_process)
        return kill_calls, process_calls

    def _info(self) -> ServerInstanceInfo:
        return ServerInstanceInfo(
            "VSharp", 35012, "ws://0.0.0.0:35012/gameServer", 1234
        )

    def test_os_kill_missing_process(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        procs = [1234]
        kill_calls, process_calls = self._setup(
            monkeypatch, procs, kill_error=ProcessLookupError()
        )

        with caplog.at_level(logging.WARNING):
            launch_servers_module.kill_server(self._info())

        assert kill_calls == [(1234, signal.SIGKILL)]
        assert process_calls == []
        assert procs == []
        assert "the process doesn't exist" in caplog.text

    def test_psutil_missing_process(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        procs = [1234]
        kill_calls, _ = self._setup(
            monkeypatch, procs, process_error=psutil.NoSuchProcess(1234)
        )

        with caplog.at_level(logging.WARNING):
            launch_servers_module.kill_server(self._info())

        assert kill_calls == [(1234, signal.SIGKILL)]
        assert procs == []
        assert "the process doesn't exist" in caplog.text

    def test_already_dead_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        procs = [1234]
        kill_calls, process_calls = self._setup(
            monkeypatch, procs, status=psutil.STATUS_ZOMBIE
        )

        launch_servers_module.kill_server(self._info())

        assert kill_calls == [(1234, signal.SIGKILL)]
        assert process_calls == [1234]
        assert procs == []

    def test_never_dies_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        procs = [1234]
        self._setup(monkeypatch, procs, status=psutil.STATUS_RUNNING)

        with pytest.raises(RuntimeError, match="could not be killed"):
            launch_servers_module.kill_server(self._info())


class TestKillProcess:
    def test_kills_and_removes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        procs = [1234, 5678]
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "PROCS", procs, raising=False)
        kill_calls: list[tuple] = []
        monkeypatch.setattr(
            launch_servers_module.os,
            "kill",
            lambda pid, sig: kill_calls.append((pid, sig)),
        )

        launch_servers_module.kill_process(1234)

        assert kill_calls == [(1234, signal.SIGKILL)]
        assert procs == [5678]


class TestServerManager:
    def test_enter_queues_exit_kills(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue = Queue()
        procs = [111, 222]
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "PROCS", procs, raising=False)
        killed: list[int] = []
        monkeypatch.setattr(
            launch_servers_module, "kill_process", lambda pid: killed.append(pid)
        )

        with launch_servers_module.server_manager(queue, 2):
            queued = [queue.get_nowait() for _ in range(2)]
            assert queued == [
                ServerInstanceInfo("_", 0, "None", pid=Undefined) for _ in range(2)
            ]

        assert killed == [111, 222]
        assert launch_servers_module.PROCS == []
