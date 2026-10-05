"""Unit tests for the broker HTTP calls, with ``httplib2.Http`` patched.

The fake transport records every request so the tests can assert the URL,
method and body without talking to the broker.
"""

import pytest
from common.validation_coverage.svm_info import SVMInfo
from config import WebsocketSourceLinks
from connection.broker_conn import requests as broker_requests
from connection.broker_conn.classes import ServerInstanceInfo
from connection.broker_conn.requests import acquire_instance, return_instance

pytestmark = [pytest.mark.unit, pytest.mark.serial]


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status


class FakeHttp:
    def __init__(self, status: int = 200, content: bytes = b"{}") -> None:
        self._status = status
        self._content = content
        self.calls: list[tuple[str, str, str | None]] = []

    def request(
        self, url: str, method: str = "GET", body: str | None = None
    ) -> tuple[FakeResponse, bytes]:
        self.calls.append((url, method, body))
        return FakeResponse(self._status), self._content


def _svm_info() -> SVMInfo:
    return SVMInfo(
        name="svm",
        launch_command="run",
        server_working_dir="/tmp",
        min_port=1,
        max_port=2,
    )


def _instance() -> ServerInstanceInfo:
    return ServerInstanceInfo(svm_name="svm", port=4000, ws_url="ws://host", pid=7)


def test_acquire_instance_gets_the_configured_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # acquire_instance then double-decodes the 200 body (#573), so the decode
    # raises; this test pins the request it issues and documents the defect.
    http = FakeHttp(content=_instance().to_json().encode("utf-8"))
    monkeypatch.setattr(broker_requests.httplib2, "Http", lambda: http)

    with pytest.raises(TypeError, match="not dict"):
        acquire_instance(_svm_info())

    url, method, body = http.calls[0]
    assert url.startswith(f"{WebsocketSourceLinks.GET_WS}?")
    assert "name=svm" in url
    assert method == "GET"
    assert body is None


def test_acquire_instance_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHttp(status=500, content=b"boom")
    monkeypatch.setattr(broker_requests.httplib2, "Http", lambda: http)

    with pytest.raises(RuntimeError, match="Not ok response"):
        acquire_instance(_svm_info())


def test_return_instance_posts_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHttp(status=200)
    monkeypatch.setattr(broker_requests.httplib2, "Http", lambda: http)

    return_instance(_instance())

    url, method, body = http.calls[0]
    assert url == WebsocketSourceLinks.POST_WS
    assert method == "POST"
    assert body == _instance().to_json()


def test_return_instance_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHttp(status=404)
    monkeypatch.setattr(broker_requests.httplib2, "Http", lambda: http)

    with pytest.raises(RuntimeError, match="Not ok response"):
        return_instance(_instance())
