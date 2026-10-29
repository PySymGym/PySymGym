import logging
import time
from contextlib import contextmanager, suppress

import psutil
import websocket
from common.config.broker_config import BrokerConfig
from common.config.connector_config import GameServerConnectorConfig
from common.validation_coverage.svm_info import SVMInfo
from connection.broker_conn.classes import ServerInstanceInfo
from connection.broker_conn.requests import acquire_instance, return_instance
from connection.errors_connection import ProcessStoppedError


@contextmanager
def process_running(pid):
    # A None pid means no process to monitor (the restart-disabled
    # placeholder): skip the check instead of letting
    # psutil.pid_exists(None) raise TypeError.
    if pid is not None and not psutil.pid_exists(pid):
        raise ProcessStoppedError
    yield


def wait_for_connection(
    server_instance: ServerInstanceInfo, connector_config: GameServerConnectorConfig
):
    ws = websocket.WebSocket()

    retries_left = connector_config.wait_for_socket_reconnection_max_retries

    while retries_left:
        with (
            suppress(
                ConnectionRefusedError,
                ConnectionResetError,
                websocket.WebSocketTimeoutException,
            ),
            process_running(server_instance.pid),
        ):
            ws.settimeout(connector_config.create_connection_timeout_sec)
            ws.connect(
                server_instance.ws_url,
                skip_utf8_validation=connector_config.skip_utf_validation,
            )
        if ws.connected:
            return ws
        time.sleep(connector_config.create_connection_timeout_sec)
        logging.info(
            f"Try connecting to {server_instance.ws_url}, {retries_left} attempts left; {server_instance}"
        )
        retries_left -= 1
    raise RuntimeError(
        f"Retries exhausted when trying to connect to {server_instance.ws_url}: {retries_left} left"
    )


@contextmanager
def game_server_socket_manager(
    svm_info: SVMInfo,
    connector_config: GameServerConnectorConfig,
    broker_config: BrokerConfig,
):
    server_instance = acquire_instance(svm_info, broker_config)
    try:
        socket = wait_for_connection(server_instance, connector_config)
        try:
            socket.settimeout(connector_config.response_timeout_sec)
            yield socket
        finally:
            socket.close()
    finally:
        return_instance(server_instance, broker_config)
