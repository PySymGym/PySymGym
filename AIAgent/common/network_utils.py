import socket

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
