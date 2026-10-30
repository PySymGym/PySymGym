"""Unit tests for the free-port helpers, driven by a fake socket.

No real port is bound: ``socket.socket`` is replaced by a factory whose sockets
fail ``bind`` for a chosen set of ports. The tests assert the selection loop,
the range bounds, the lock usage and the retry/exhaustion behaviour.
"""

import inspect

import pytest
from common import network_utils
from common.network_utils import look_for_free_port_locked, next_free_port
from common.validation_coverage.svm_info import SVMInfo

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def test_next_free_port_requires_an_explicit_range_and_address() -> None:
    # The port range and the probe address are the caller's policy: no
    # broker-derived defaults (they used to import the script-level config
    # into the shared layer). The address is required so every call site
    # states the interface it probes — the one the real bind will use.
    params = inspect.signature(next_free_port).parameters
    assert list(params) == ["min_port", "max_port", "address"]
    assert all(p.default is inspect.Parameter.empty for p in params.values())


class FakeSocket:
    def __init__(self, occupied_ports: set[int]) -> None:
        self._occupied_ports = occupied_ports
        self.bound_address: tuple[str, int] | None = None
        self.closed = False
        self.listened = False

    def bind(self, address: tuple[str, int]) -> None:
        port = address[1]
        if port in self._occupied_ports:
            raise OSError(f"port {port} is busy")
        self.bound_address = address

    def listen(self, backlog: int) -> None:
        self.listened = True

    def close(self) -> None:
        self.closed = True


class FakeSocketFactory:
    def __init__(self, occupied_ports: set[int]) -> None:
        self._occupied_ports = occupied_ports
        self.created: list[FakeSocket] = []

    def __call__(self, family: int, type: int) -> FakeSocket:
        sock = FakeSocket(self._occupied_ports)
        self.created.append(sock)
        return sock


class FakeLock:
    def __init__(self) -> None:
        self.entries = 0

    def __enter__(self) -> "FakeLock":
        self.entries += 1
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _svm_info(min_port: int = 5, max_port: int = 8) -> SVMInfo:
    return SVMInfo(
        name="svm",
        launch_command="run",
        server_working_dir="/tmp",
        min_port=min_port,
        max_port=max_port,
    )


def test_next_free_port_returns_first_free_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeSocketFactory(occupied_ports={5})
    monkeypatch.setattr(network_utils.socket, "socket", factory)

    assert next_free_port(5, 8, "localhost") == 6
    assert len(factory.created) == 1
    assert factory.created[0].closed is True


def test_next_free_port_probes_the_callers_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The probe must bind the interface the real bind will use: the fake
    # records the bound address, so the assertion is on the probe socket.
    factory = FakeSocketFactory(occupied_ports={5})
    monkeypatch.setattr(network_utils.socket, "socket", factory)

    assert next_free_port(5, 8, "127.0.0.1") == 6
    assert factory.created[0].bound_address == ("127.0.0.1", 6)


def test_next_free_port_starts_at_min_port(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeSocketFactory(occupied_ports=set())
    monkeypatch.setattr(network_utils.socket, "socket", factory)

    assert next_free_port(10, 12, "localhost") == 10


def test_next_free_port_raises_when_range_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeSocketFactory(occupied_ports={5, 6, 7})
    monkeypatch.setattr(network_utils.socket, "socket", factory)

    with pytest.raises(IOError, match="no free ports"):
        next_free_port(5, 7, "localhost")

    # The probe socket is closed on the error path as well.
    assert factory.created[0].closed is True


def test_next_free_port_closes_the_socket_for_an_empty_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeSocketFactory(occupied_ports=set())
    monkeypatch.setattr(network_utils.socket, "socket", factory)

    with pytest.raises(IOError, match="no free ports"):
        next_free_port(8, 5, "localhost")

    assert factory.created[0].closed is True


def test_look_for_free_port_locked_binds_and_listens_under_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeSocketFactory(occupied_ports=set())
    monkeypatch.setattr(network_utils.socket, "socket", factory)
    lock = FakeLock()

    port, server_socket = look_for_free_port_locked(lock, _svm_info(5, 8))

    assert port == 5
    assert lock.entries == 1
    assert server_socket is factory.created[1]
    assert server_socket.bound_address == ("localhost", 5)
    assert server_socket.listened is True
    # The probe and the held server socket bind the same interface: a
    # port free on one interface is not necessarily free on the other.
    assert factory.created[0].bound_address == ("localhost", 5)


def test_look_for_free_port_locked_uses_the_given_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeSocketFactory(occupied_ports=set())
    monkeypatch.setattr(network_utils.socket, "socket", factory)
    lock = FakeLock()

    port, server_socket = look_for_free_port_locked(
        lock, _svm_info(5, 8), address="127.0.0.1"
    )

    assert port == 5
    assert factory.created[0].bound_address == ("127.0.0.1", 5)
    assert server_socket.bound_address == ("127.0.0.1", 5)


def test_look_for_free_port_locked_raises_without_attempts() -> None:
    with pytest.raises(RuntimeError, match="Failed to occupy port"):
        look_for_free_port_locked(FakeLock(), _svm_info(), attempts=0)


def test_look_for_free_port_locked_retries_then_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def busy(min_port: int, max_port: int, address: str) -> int:
        calls.append(address)
        raise OSError("busy")

    monkeypatch.setattr(network_utils, "next_free_port", busy)

    with pytest.raises(RuntimeError, match="Failed to occupy port"):
        look_for_free_port_locked(FakeLock(), _svm_info(), attempts=3)

    assert len(calls) == 3
    # The retry loop forwards the address on every attempt.
    assert calls == ["localhost"] * 3
