import logging
import socket
from typing import Tuple

from common.validation_coverage.svm_info import SVMInfo
from config import BrokerConfig


def next_free_port(
    min_port=BrokerConfig.BROKER_PORT + 1,
    max_port=BrokerConfig.BROKER_PORT + 1000,
):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while min_port <= max_port:
        try:
            sock.bind(("", min_port))
            sock.close()
            return min_port
        except OSError:
            min_port += 1
    raise IOError("no free ports")


def look_for_free_port_locked(
    lock, svm_info: SVMInfo, attempts=100
) -> Tuple[int, socket.socket]:
    if attempts <= 0:
        raise RuntimeError("Failed to occupy port")
    logging.debug(f"Looking for port... Attempls left: {attempts}.")
    try:
        with lock:
            port = next_free_port(svm_info.min_port, svm_info.max_port)  # type: ignore
            logging.debug(f"Try to occupy {port=}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind(
                ("localhost", port)
            )  # TODO: working within a local network
            server_socket.listen(1)
            return port, server_socket
    except OSError:
        logging.debug("Failed to occupy port")
        return look_for_free_port_locked(lock, svm_info, attempts - 1)
