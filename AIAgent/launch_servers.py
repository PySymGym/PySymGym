import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from queue import Empty, Queue

import psutil
import yaml
from aiohttp import web
from common.network_utils import next_free_port
from common.config.config import Config
from common.validation_coverage.svm_info import SVMInfo
from config import BrokerConfig, FeatureConfig, GeneralConfig
from connection.broker_conn.classes import (
    ServerInstanceInfo,
    Undefined,
    WSUrl,
)

routes = web.RouteTableDef()

FAILED_TO_INSTANTIATE_ERROR = "TCP server failed"
avoid_same_free_port_lock = asyncio.Lock()


class InstanceIssuanceError(RuntimeError):
    """The broker cannot fulfil a /get_ws request.

    Raised when a live instance is dequeued while the restart feature is
    disabled (the queue must hold placeholders only) or when a freshly
    started game server does not open its port within the startup timeout.
    """


async def wait_for_port(port, timeout=10):
    start_time = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        while time.time() - start_time < timeout:
            if sock.connect_ex(("", port)) == 0:
                return True
            await asyncio.sleep(0.1)
        return False
    finally:
        sock.close()


@routes.get("/get_ws")
async def dequeue_instance(request):
    try:
        server_info = SERVER_INSTANCES.get(block=False)
        # The queue must hold placeholders only: a live instance is
        # re-queued only when the restart feature is disabled, and
        # re-issuing it would hand the client a server it does not own.
        # A typed error (not an assert): it survives python -O and the
        # except-Exception path below re-queues the instance and returns
        # 500 instead of a silent ws_url="None".
        if server_info.pid is not Undefined:
            raise InstanceIssuanceError(
                f"cannot issue {server_info}: a live instance was requeued, "
                "which only happens with ON_GAME_SERVER_RESTART disabled; "
                "the queue must hold placeholders only"
            )
        server_info = await run_server_instance(
            SVMInfo(**request.query),
            should_start_server=FeatureConfig.ON_GAME_SERVER_RESTART.enabled,
        )
        logging.info(f"issued {server_info}: {psutil.Process(server_info.pid)}")
        return web.json_response(server_info.to_dict())
    except Empty:
        logging.error("Couldn't dequeue instance, the queue is not replenishing")
        raise
    except Exception as e:
        logging.error(e, exc_info=True)
        SERVER_INSTANCES.put(server_info)
        raise


@routes.post("/post_ws")
async def enqueue_instance(request):
    returned_instance_info_raw = await request.read()
    returned_instance_info = ServerInstanceInfo.from_json(
        returned_instance_info_raw.decode("utf-8")
    )
    logging.info(f"got {returned_instance_info} from client")

    if FeatureConfig.ON_GAME_SERVER_RESTART.enabled:
        await kill_server(returned_instance_info)
        returned_instance_info = ServerInstanceInfo(
            returned_instance_info.svm_name,
            returned_instance_info.port,
            returned_instance_info.ws_url,
            pid=Undefined,
        )

    SERVER_INSTANCES.put(returned_instance_info)
    logging.info(f"enqueue {returned_instance_info}")
    return web.HTTPOk()


@routes.post("/send_res")
async def append_results(request):
    global RESULTS
    data = await request.read()
    decoded = data.decode("utf-8")
    RESULTS.append(decoded)
    return web.HTTPOk()


@routes.get("/recv_res")
async def send_and_clear_results(request):
    global RESULTS
    if not RESULTS:
        raise RuntimeError("Must play a game first")
    rst = json.dumps(RESULTS)
    RESULTS = []
    return web.Response(text=rst)


def get_socket_url(port: int) -> WSUrl:
    return f"ws://0.0.0.0:{port}/gameServer"


async def run_server_instance(
    svm_info: SVMInfo, should_start_server: bool
) -> ServerInstanceInfo:
    svm_name = svm_info.name

    def launcher(port: int) -> str:
        return svm_info.launch_command.format(port=port)

    min_port = svm_info.min_port
    max_port = svm_info.max_port
    server_working_dir = svm_info.server_working_dir

    if not should_start_server:
        return ServerInstanceInfo(svm_name, 0, "None", pid=Undefined)

    def start_server() -> tuple[subprocess.Popen[bytes], int]:
        port = next_free_port(min_port, max_port)
        launch_server = launcher(port)
        proc = subprocess.Popen(
            launch_server.split(),
            # The pipe was never read: a server writing more than the pipe
            # buffer blocked its game until the timeout. DEVNULL discards
            # the output (nothing consumed it) and cannot fill up.
            stdout=subprocess.DEVNULL,
            start_new_session=True,
            cwd=Path(server_working_dir).absolute(),
            # Pin the .NET runtime to the major.minor the game server's
            # runtimeconfig.json declares (.NET 8 — the runtime the dataset
            # was generated with, the same guarantee CI has): LatestPatch
            # takes the highest installed patch of that major.minor and
            # never rolls forward to a newer major, so a mismatched ambient
            # runtime fails loudly at server start instead of silently
            # changing the path-condition encoding (#669). The variable is
            # inert for non-.NET launch commands.
            env={
                **os.environ,
                "DOTNET_ROLL_FORWARD": FeatureConfig.DOTNET_ROLL_FORWARD,
            },
        )
        logging.info("bash exec cmd: " + launch_server)
        return proc, port

    async with avoid_same_free_port_lock:
        proc, port = start_server()
        # Track the handle from birth: the pid alone cannot be reaped.
        PROCS[proc.pid] = proc
        if not await wait_for_port(port):
            # Fail fast: a server that never opens its port would make the
            # client retry a dead port for the whole connection timeout.
            # The kill runs in a worker thread so the async handler does
            # not block on the reap.
            await asyncio.to_thread(kill_process, proc.pid)
            raise InstanceIssuanceError(
                f"game server did not open port {port} within the startup timeout"
            )

    launch_server = launcher(port)
    logging.info(f"running new instance on {port=} with {proc.pid=}:" + launch_server)

    ws_url = get_socket_url(port)
    return ServerInstanceInfo(svm_name, port, ws_url, proc.pid)


async def run_servers(count: int) -> list[ServerInstanceInfo]:
    servers_start_tasks = []

    async def run():
        server_info = ServerInstanceInfo("_", 0, "None", pid=Undefined)
        servers_start_tasks.append(server_info)

    await asyncio.gather(*[run() for _ in range(count)])

    return servers_start_tasks


def _tracked_handle(pid: int) -> subprocess.Popen[bytes] | None:
    """Look up the tracked handle for ``pid``.

    An untracked pid (already reaped, or never registered) is a no-op, not
    an error: log the warning and return ``None``.
    """
    proc = PROCS.get(pid)
    if proc is None:
        logging.warning(
            f"Failed to kill the process with ID={pid}: the process doesn't exist"
        )
    return proc


def _kill_process_group(pid: int) -> None:
    """SIGKILL the whole process group of ``pid``.

    ``start_new_session=True`` makes the child the leader of its own session
    and process group, so ``os.getpgid(pid) == pid`` and the signal reaches
    the grandchild dotnet processes, not just the direct child. A vanished
    group is not an error: the caller still reaps the handle.
    """
    with suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


async def kill_server(server_instance: ServerInstanceInfo) -> None:
    pid = server_instance.pid
    proc = _tracked_handle(pid)
    if proc is None:
        return
    _kill_process_group(pid)
    # Reap with proc.wait() (POSIX waitpid): a zombie is collected, unlike
    # the old psutil-status poll that treated STATUS_ZOMBIE as "killed" and
    # left the zombie behind. The blocking wait runs in a worker thread so
    # the event loop (and every other broker request) is not stalled; the
    # budget is the existing restart-feature retry values. The handle is
    # dropped only once the death is confirmed: a process that survived the
    # kill stays tracked so the shutdown loop still reaches it.
    timeout = (
        FeatureConfig.ON_GAME_SERVER_RESTART.wait_for_reset_retries
        * FeatureConfig.ON_GAME_SERVER_RESTART.wait_for_reset_time
    )
    try:
        await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=timeout)
    except TimeoutError as err:
        raise RuntimeError(f"{server_instance} could not be killed") from err
    PROCS.pop(pid, None)


def kill_process(pid: int) -> None:
    proc = _tracked_handle(pid)
    if proc is None:
        return
    _kill_process_group(pid)
    proc.wait()
    PROCS.pop(pid, None)


@contextmanager
def server_manager(server_queue: Queue[ServerInstanceInfo], server_count: int):
    servers_info = asyncio.run(run_servers(server_count))

    for server_info in servers_info:
        server_queue.put(server_info)
    try:
        yield
    finally:
        # A per-pid failure (an already-exited server) must not stop the
        # loop: the remaining servers would be orphaned.
        for pid in list(PROCS):
            try:
                kill_process(pid)
            except Exception:
                logging.exception(f"Failed to kill server {pid} during shutdown")
        PROCS.clear()


def main(config: str):
    global SERVER_INSTANCES, PROCS, RESULTS

    logging.basicConfig(
        level=GeneralConfig.LOGGER_LEVEL,
        filename="instance_manager.log",
        filemode="w",
        format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
    )

    # Queue[ServerInstanceInfo]
    SERVER_INSTANCES = Queue()
    PROCS = {}
    RESULTS = []

    with open(config, "r") as file:
        trainings_parameters = yaml.safe_load(file)
    config: Config = Config(**trainings_parameters)
    server_count = config.validation_config.validation_mode.process_count

    with server_manager(SERVER_INSTANCES, server_count):
        app = web.Application()
        app.add_routes(routes)
        web.run_app(app, port=BrokerConfig.BROKER_PORT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch servers using configuration from a .yml file."
    )

    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )

    args = parser.parse_args()
    config = args.config
    main(config)
