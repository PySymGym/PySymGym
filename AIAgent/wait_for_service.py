"""Wait for a local service to become ready (stdlib only).

The e2e workflow starts the MLflow server and the game-server broker as
background processes inside the step that uses them; readiness is polled
instead of a fixed sleep, which would race with the variable startup time.
This script is the single implementation of that polling for both service
kinds: an HTTP health endpoint (MLflow) or a TCP port (the broker).

Usage:
    python3 wait_for_service.py http --url http://127.0.0.1:8080/health --name "MLflow server"
    python3 wait_for_service.py tcp --host 127.0.0.1 --port 35000 --name "game-server broker"

Exits non-zero with "<name> did not become ready on <endpoint>" when the
deadline passes without success.
"""

import argparse
import socket
import time
import urllib.request


def wait_for_http(url: str, timeout_sec: float, poll_interval_sec: float = 2.0) -> None:
    """Poll an HTTP endpoint until it answers with status 200.

    Parameters
    ----------
    url : str
        Health endpoint to poll.
    timeout_sec : float
        Total time to keep polling before giving up.
    poll_interval_sec : float, optional
        Pause between attempts (default 2.0).

    Raises
    ------
    TimeoutError
        If the endpoint did not answer with 200 within ``timeout_sec``.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(url)
        time.sleep(poll_interval_sec)


def wait_for_tcp(
    host: str, port: int, timeout_sec: float, poll_interval_sec: float = 2.0
) -> None:
    """Poll a TCP port until a connection succeeds.

    Parameters
    ----------
    host : str
        Hostname or address to connect to.
    port : int
        Port to connect to.
    timeout_sec : float
        Total time to keep polling before giving up.
    poll_interval_sec : float, optional
        Pause between attempts (default 2.0).

    Raises
    ------
    TimeoutError
        If no connection succeeded within ``timeout_sec``.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"tcp://{host}:{port}")
        time.sleep(poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait until a local HTTP health endpoint or TCP port is ready."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    http_parser = subparsers.add_parser("http", help="wait for an HTTP health endpoint")
    http_parser.add_argument("--url", required=True, help="health endpoint to poll")

    tcp_parser = subparsers.add_parser("tcp", help="wait for a TCP port")
    tcp_parser.add_argument("--host", required=True, help="host to connect to")
    tcp_parser.add_argument(
        "--port", type=int, required=True, help="port to connect to"
    )

    for subparser in (http_parser, tcp_parser):
        subparser.add_argument(
            "--name", default="service", help="service name used in the failure message"
        )
        subparser.add_argument(
            "--timeout", type=float, default=120.0, help="total seconds to keep polling"
        )
        subparser.add_argument(
            "--interval", type=float, default=2.0, help="seconds between attempts"
        )

    args = parser.parse_args()
    try:
        if args.mode == "http":
            wait_for_http(args.url, args.timeout, args.interval)
        else:
            wait_for_tcp(args.host, args.port, args.timeout, args.interval)
    except TimeoutError as error:
        raise SystemExit(f"{args.name} did not become ready on {error}") from None


if __name__ == "__main__":
    main()
