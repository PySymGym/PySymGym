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


def test_acquire_instance_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHttp(status=500, content=b"boom")
    monkeypatch.setattr(broker_requests.httplib2, "Http", lambda: http)

    with pytest.raises(RuntimeError, match="Not ok response"):
        acquire_instance(_svm_info())

    # The request is issued before the response is decoded, so the non-200
    # path pins the request shape without touching the #573 decode.
    url, method, body = http.calls[0]
    assert url.startswith(f"{WebsocketSourceLinks.GET_WS}?")
    assert "name=svm" in url
    assert "launch_command=run" in url
    assert "min_port=1" in url
    assert "max_port=2" in url
    assert method == "GET"
    assert body is None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#573: acquire_instance double-decodes the broker response — "
        "ServerInstanceInfo.from_json(json.loads(...)) raises TypeError on "
        "the decoded dict. #573's fix spec: from_dict(json.loads(...)) (or "
        "pass the raw content bytes to from_json)."
    ),
)
def test_acquire_instance_decodes_single_encoded_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The broker's /get_ws single-encodes (launch_servers.py, pinned by
    # C1's test_success_issues_a_real_instance), so the wire format is a
    # single-encoded JSON object. This test feeds the client that
    # single-encoded body; the client's double-decode raises TypeError on
    # the decoded dict, so it xfails until #573's client fix lands. The
    # success-path lines are excluded (see docs/testing.rst).
    http = FakeHttp(content=_instance().to_json().encode("utf-8"))
    monkeypatch.setattr(broker_requests.httplib2, "Http", lambda: http)

    acquired = acquire_instance(_svm_info())

    assert acquired == _instance()


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
