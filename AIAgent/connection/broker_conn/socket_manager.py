import logging
import time
from contextlib import contextmanager, suppress

import psutil
import websocket
from common.validation_coverage.svm_info import SVMInfo
from config import GameServerConnectorConfig
from connection.broker_conn.classes import ServerInstanceInfo
from connection.broker_conn.requests import acquire_instance, return_instance
from connection.errors_connection import ProcessStoppedError


@contextmanager
def process_running(pid):
    if not psutil.pid_exists(pid):
        raise ProcessStoppedError
    yield


def wait_for_connection(server_instance: ServerInstanceInfo):
    ws = websocket.WebSocket()

    retries_left = GameServerConnectorConfig.WAIT_FOR_SOCKET_RECONNECTION_MAX_RETRIES

    while retries_left:
        with (
            suppress(
                ConnectionRefusedError,
                ConnectionResetError,
                websocket.WebSocketTimeoutException,
            ),
            process_running(server_instance.pid),
        ):
            ws.settimeout(GameServerConnectorConfig.CREATE_CONNECTION_TIMEOUT_SEC)
            ws.connect(
                server_instance.ws_url,
                skip_utf8_validation=GameServerConnectorConfig.SKIP_UTF_VALIDATION,
            )
        if ws.connected:
            return ws
        time.sleep(GameServerConnectorConfig.CREATE_CONNECTION_TIMEOUT_SEC)
        logging.info(
            f"Try connecting to {server_instance.ws_url}, {retries_left} attempts left; {server_instance}"
        )
        retries_left -= 1
    raise RuntimeError(
        f"Retries exhausted when trying to connect to {server_instance.ws_url}: {retries_left} left"
    )


@contextmanager
def game_server_socket_manager(svm_info: SVMInfo):
    server_instance = acquire_instance(svm_info)
    try:
        socket = wait_for_connection(server_instance)
        try:
            socket.settimeout(GameServerConnectorConfig.RESPONCE_TIMEOUT_SEC)
            yield socket
        finally:
            socket.close()
    finally:
        return_instance(server_instance)
