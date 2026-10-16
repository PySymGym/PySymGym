"""Tests for the service-readiness polling used by the e2e workflow."""

import http.server
import socket
import sys
import threading

import pytest

from wait_for_service import main, wait_for_http, wait_for_tcp


class _Handler(http.server.BaseHTTPRequestHandler):
    status_code = 200

    def do_GET(self):
        self.send_response(self.status_code)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def _serve(handler_class):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestWaitForTcp:
    def test_returns_when_port_accepts(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            wait_for_tcp(
                "127.0.0.1",
                listener.getsockname()[1],
                timeout_sec=5,
                poll_interval_sec=0.05,
            )
        finally:
            listener.close()

    def test_times_out_on_closed_port(self):
        port = _free_port()
        with pytest.raises(TimeoutError, match=f"tcp://127.0.0.1:{port}"):
            wait_for_tcp("127.0.0.1", port, timeout_sec=0.3, poll_interval_sec=0.05)


class _Handler500(_Handler):
    status_code = 500


class TestWaitForHttp:
    def test_returns_on_200(self):
        server, port = _serve(_Handler)
        try:
            wait_for_http(
                f"http://127.0.0.1:{port}/health", timeout_sec=5, poll_interval_sec=0.05
            )
        finally:
            server.shutdown()

    def test_non_200_is_not_ready(self):
        server, port = _serve(_Handler500)
        try:
            with pytest.raises(TimeoutError, match=f"http://127.0.0.1:{port}/health"):
                wait_for_http(
                    f"http://127.0.0.1:{port}/health",
                    timeout_sec=0.3,
                    poll_interval_sec=0.05,
                )
        finally:
            server.shutdown()

    def test_times_out_on_closed_port(self):
        port = _free_port()
        with pytest.raises(TimeoutError, match=f"http://127.0.0.1:{port}/health"):
            wait_for_http(
                f"http://127.0.0.1:{port}/health",
                timeout_sec=0.3,
                poll_interval_sec=0.05,
            )


class TestMain:
    def test_failure_message_carries_service_name(self, monkeypatch):
        port = _free_port()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "wait_for_service.py",
                "tcp",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--name",
                "game-server broker",
                "--timeout",
                "0.3",
                "--interval",
                "0.05",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert (
            str(excinfo.value)
            == f"game-server broker did not become ready on tcp://127.0.0.1:{port}"
        )

    def test_success_exits_cleanly(self, monkeypatch):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "wait_for_service.py",
                    "tcp",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(listener.getsockname()[1]),
                    "--timeout",
                    "5",
                    "--interval",
                    "0.05",
                ],
            )
            main()
        finally:
            listener.close()
