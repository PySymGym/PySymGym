"""Entrypoint seam tests for ``launch_servers.py``.

The broker is driven with fakes: a scripted socket for ``wait_for_port``,
fake requests for the aiohttp handlers, a fake ``Popen`` for the server
lifecycle, and fake ``os.killpg``/``os.getpgid`` plus the ``fake_proc``
stand-in for the kill paths. No real processes, ports, or event-loop-bound
state are touched.
"""

import asyncio
import json
import logging
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace

import pytest
import yaml

import launch_servers as launch_servers_module
from common.validation_coverage.svm_info import SVMInfo
from config import BrokerConfig, FeatureConfig, OnGameServerRestartFeature
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
        # The response body is the single-encoded JSON of the issued info.
        assert json.loads(response.text) == issued.to_dict()
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

        async def fake_kill_server(info):
            kill_calls.append(info)

        monkeypatch.setattr(launch_servers_module, "kill_server", fake_kill_server)
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
        popen_instances: list = []

        class FakePopen:
            # The nested start_server annotates subprocess.Popen[bytes]; the
            # annotation is evaluated at def time, so the fake must subscript.
            def __class_getitem__(cls, item):
                return cls

            def __init__(self, args, **kwargs):
                popen_calls.append((args, kwargs))
                popen_instances.append(self)
                self.pid = 4321

        monkeypatch.setattr(launch_servers_module.subprocess, "Popen", FakePopen)

        async def fake_wait_for_port(port, timeout=10):
            return True

        monkeypatch.setattr(launch_servers_module, "wait_for_port", fake_wait_for_port)
        procs: dict = {}
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
            "stdout": subprocess.DEVNULL,
            "start_new_session": True,
            "cwd": Path(tmp_path).absolute(),
        }
        # The Popen handle is tracked by pid, not the pid alone.
        assert procs == {4321: popen_instances[0]}
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
        procs: dict,
        *,
        killpg_error=None,
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
        killpg_calls: list[tuple] = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))
            if killpg_error is not None:
                raise killpg_error

        monkeypatch.setattr(launch_servers_module.os, "killpg", fake_killpg)
        getpgid_calls: list[int] = []

        def fake_getpgid(pid):
            getpgid_calls.append(pid)
            # start_new_session=True: the child's pid is its group id.
            return pid

        monkeypatch.setattr(launch_servers_module.os, "getpgid", fake_getpgid)
        return killpg_calls, getpgid_calls

    def _info(self) -> ServerInstanceInfo:
        return ServerInstanceInfo(
            "VSharp", 35012, "ws://0.0.0.0:35012/gameServer", 1234
        )

    def test_untracked_pid_warns_and_returns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        killpg_calls, getpgid_calls = self._setup(monkeypatch, {})

        with caplog.at_level(logging.WARNING):
            asyncio.run(launch_servers_module.kill_server(self._info()))

        assert killpg_calls == []
        assert getpgid_calls == []
        assert "the process doesn't exist" in caplog.text

    def test_kills_the_group_and_reaps(
        self, monkeypatch: pytest.MonkeyPatch, fake_proc
    ) -> None:
        proc = fake_proc()
        killpg_calls, getpgid_calls = self._setup(monkeypatch, {1234: proc})

        asyncio.run(launch_servers_module.kill_server(self._info()))

        assert getpgid_calls == [1234]
        assert killpg_calls == [(1234, signal.SIGKILL)]
        assert proc.waited is True
        assert launch_servers_module.PROCS == {}

    def test_vanished_group_still_reaps(
        self, monkeypatch: pytest.MonkeyPatch, fake_proc
    ) -> None:
        proc = fake_proc()
        self._setup(monkeypatch, {1234: proc}, killpg_error=ProcessLookupError())

        asyncio.run(launch_servers_module.kill_server(self._info()))

        assert proc.waited is True
        assert launch_servers_module.PROCS == {}

    def test_never_dies_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class HangingProcess:
            pid = 1234

            def wait(self) -> int:
                time.sleep(0.2)
                return 0

        proc = HangingProcess()
        self._setup(monkeypatch, {1234: proc})

        with pytest.raises(RuntimeError, match="could not be killed"):
            asyncio.run(launch_servers_module.kill_server(self._info()))

        # The process survived the kill, so it stays tracked: the shutdown
        # loop (or a later post_ws) still reaches it.
        assert launch_servers_module.PROCS == {1234: proc}


class TestKillProcess:
    def test_kills_the_group_and_reaps(
        self, monkeypatch: pytest.MonkeyPatch, fake_proc
    ) -> None:
        proc = fake_proc()
        other = fake_proc(pid=5678)
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(
            launch_servers_module, "PROCS", {1234: proc, 5678: other}, raising=False
        )
        killpg_calls: list[tuple] = []
        monkeypatch.setattr(
            launch_servers_module.os,
            "killpg",
            lambda pgid, sig: killpg_calls.append((pgid, sig)),
        )
        monkeypatch.setattr(launch_servers_module.os, "getpgid", lambda pid: pid)

        launch_servers_module.kill_process(1234)

        assert killpg_calls == [(1234, signal.SIGKILL)]
        assert proc.waited is True
        assert other.waited is False
        assert launch_servers_module.PROCS == {5678: other}

    def test_untracked_pid_warns_and_returns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "PROCS", {}, raising=False)
        killpg_calls: list[tuple] = []
        monkeypatch.setattr(
            launch_servers_module.os,
            "killpg",
            lambda pgid, sig: killpg_calls.append((pgid, sig)),
        )

        with caplog.at_level(logging.WARNING):
            launch_servers_module.kill_process(1234)

        assert killpg_calls == []
        assert "the process doesn't exist" in caplog.text


class TestServerManager:
    def test_enter_queues_exit_kills(
        self, monkeypatch: pytest.MonkeyPatch, fake_proc
    ) -> None:
        queue = Queue()
        procs = {111: fake_proc(pid=111), 222: fake_proc(pid=222)}
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
        assert launch_servers_module.PROCS == {}

    def test_shutdown_survives_per_pid_errors(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        queue = Queue()
        procs = {111: object(), 222: object()}
        # The global is only created by main(); set it fresh for the test.
        monkeypatch.setattr(launch_servers_module, "PROCS", procs, raising=False)
        killed: list[int] = []

        def flaky_kill_process(pid):
            if pid == 111:
                raise ProcessLookupError()
            killed.append(pid)

        monkeypatch.setattr(launch_servers_module, "kill_process", flaky_kill_process)

        with caplog.at_level(logging.ERROR):
            with launch_servers_module.server_manager(queue, 2):
                pass

        # The first pid's failure did not stop the loop: the second server
        # was still killed, and PROCS was cleared.
        assert killed == [222]
        assert launch_servers_module.PROCS == {}
        assert "Failed to kill server 111 during shutdown" in caplog.text


def launch_servers_main_config() -> dict:
    """A production-shaped config (alias keys) for the ``main`` tests."""
    return {
        "OptunaConfig": {
            "n_startup_trials": 1,
            "n_trials": 1,
            "n_jobs": 1,
            "study_direction": "maximize",
        },
        "TrainingConfig": {
            "dynamic_dataset": False,
            "train_percentage": 1.0,
            "threshold_coverage": 100,
            "load_to_cpu": False,
            "epochs": 1,
            "seed": 99,
        },
        "ValidationConfig": {
            "validation_mode": {
                "val_type": "svms_each_step",
                "process_count": 2,
                "fail_immediately": True,
                "PlatformsConfig": [
                    {
                        "name": "dotnet",
                        "DatasetConfigs": [
                            {
                                "dataset_base_path": "/dummy",
                                "dataset_description": "/dummy.json",
                            }
                        ],
                        "SVMSInfo": [
                            {
                                "name": "VSharp",
                                "launch_command": "run --port {port}",
                                "min_port": 35000,
                                "max_port": 35050,
                                "server_working_dir": "/srv",
                            }
                        ],
                    }
                ],
            }
        },
        "MLFlowConfig": {"experiment_name": "CI"},
    }


class TestMain:
    """``launch_servers.main``: config loading and broker startup."""

    def test_main(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # chdir so the instance_manager.log from logging.basicConfig lands in tmp.
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.safe_dump(launch_servers_main_config()))

        sm_calls: list[tuple] = []

        @contextmanager
        def fake_server_manager(server_queue, server_count):
            sm_calls.append((server_queue, server_count))
            yield

        monkeypatch.setattr(
            launch_servers_module, "server_manager", fake_server_manager
        )
        run_app_calls: list[tuple] = []
        monkeypatch.setattr(
            launch_servers_module.web,
            "run_app",
            lambda app, **kwargs: run_app_calls.append((app, kwargs)),
        )

        launch_servers_module.main(str(config_path))

        # server_manager got the fresh queue and the configured process count.
        assert len(sm_calls) == 1
        server_queue, server_count = sm_calls[0]
        assert server_count == 2
        assert isinstance(server_queue, Queue)
        assert server_queue is launch_servers_module.SERVER_INSTANCES

        # The app carries exactly the four declared broker routes (aiohttp
        # adds implicit HEAD routes for the GETs, which are filtered out).
        (app, kwargs) = run_app_calls[0]
        assert kwargs == {"port": BrokerConfig.BROKER_PORT}
        routes = sorted(
            (route.method, route.resource.canonical)
            for route in app.router.routes()
            if route.method != "HEAD"
        )
        assert routes == [
            ("GET", "/get_ws"),
            ("GET", "/recv_res"),
            ("POST", "/post_ws"),
            ("POST", "/send_res"),
        ]

        # The module globals are reset for the run.
        assert isinstance(launch_servers_module.SERVER_INSTANCES, Queue)
        assert launch_servers_module.PROCS == {}
        assert launch_servers_module.RESULTS == []
