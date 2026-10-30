import logging
import socket
from typing import Tuple

from common.validation_coverage.svm_info import SVMInfo


def next_free_port(min_port: int, max_port: int, address: str) -> int:
    """Scan ``[min_port, max_port]`` and return the first free TCP port.

    The range and the address are the caller's policy: this module
    deliberately does not know about the broker port or any other fixed
    range, so the shared layer stays importable without the script-level
    ``config`` module.

    Parameters
    ----------
    min_port : int
        First port to try (inclusive).
    max_port : int
        Last port to try (inclusive).
    address : str
        The interface to probe. Pass the address the real bind will use:
        a port is only "free" for the caller if it can be bound on that
        interface (probing ``""`` while the server binds ``localhost``
        would report ports busy on other interfaces, and say nothing
        about the one the server actually uses).

    Returns
    -------
    int
        The first port in the range that could be bound on ``address``.

    Raises
    ------
    IOError
        If no port in the range is free.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        while min_port <= max_port:
            try:
                sock.bind((address, min_port))
                return min_port
            except OSError:
                min_port += 1
        raise IOError("no free ports")
    finally:
        sock.close()


def look_for_free_port_locked(
    lock, svm_info: SVMInfo, attempts=100, address="localhost"
) -> Tuple[int, socket.socket]:
    if attempts <= 0:
        raise RuntimeError("Failed to occupy port")
    logging.debug(f"Looking for port... Attempts left: {attempts}.")
    try:
        with lock:
            port = next_free_port(svm_info.min_port, svm_info.max_port, address)
            logging.debug(f"Try to occupy {port=}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((address, port))  # TODO: working within a local network
            server_socket.listen(1)
            return port, server_socket
    except OSError:
        logging.debug("Failed to occupy port")
        return look_for_free_port_locked(lock, svm_info, attempts - 1, address)
