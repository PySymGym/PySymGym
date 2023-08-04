import argparse
import json
import os
import signal
import subprocess
from contextlib import contextmanager
from queue import Empty, Queue

from aiohttp import web

from common.constants import SERVER_WORKING_DIR
from config import BrokerConfig, FeatureConfig, GeneralConfig, ServerConfig
from connection.broker_conn.classes import ServerInstanceInfo, Undefined, WSUrl

routes = web.RouteTableDef()


@routes.get("/get_ws")
async def dequeue_instance(request):
    try:
        server_info = SERVER_INSTANCES.get(timeout=0.1)
        print(f"issued {server_info}")
        return web.json_response(server_info.to_json())
    except Empty as e:
        print(f"{os.getpid()} tried to dequeue an empty queue. Waiting...")
        return web.Response(text=str(e))


@routes.post("/post_ws")
async def enqueue_instance(request):
    returned_instance_info_raw = await request.read()
    returned_instance_info = ServerInstanceInfo.from_json(
        returned_instance_info_raw.decode("utf-8")
    )

    print(f"put back {returned_instance_info}")
    if FeatureConfig.ON_GAME_SERVER_RESTART:
        kill_server(returned_instance_info.pid, forget=True)
        returned_instance_info = run_server_instance(
            port=returned_instance_info.port, start_server=START_SERVERS
        )

    SERVER_INSTANCES.put(returned_instance_info)
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


def run_server_instance(port: int, start_server: bool) -> ServerInstanceInfo:
    launch_server = [
        "dotnet",
        "VSharp.ML.GameServer.Runner.dll",
        "--checkactualcoverage",
        "--port",
    ]
    server_pid = Undefined
    if start_server:
        proc = subprocess.Popen(
            launch_server + [str(port)],
            start_new_session=True,
            cwd=SERVER_WORKING_DIR,
        )
        server_pid = proc.pid
        PROCS.append(server_pid)
        print(f"{server_pid}: " + " ".join(launch_server + [str(port)]))

    ws_url = get_socket_url(port)
    return ServerInstanceInfo(port, ws_url, server_pid)


def run_servers(
    num_inst: int, start_port: int, start_servers: bool
) -> list[ServerInstanceInfo]:
    servers_info = []
    for i in range(num_inst):
        server_info = run_server_instance(start_port + i, start_server=start_servers)
        servers_info.append(server_info)

    return servers_info


def kill_server(pid: int, forget):
    os.kill(pid, signal.SIGKILL)
    if forget:
        PROCS.remove(pid)
    print(f"killed {pid}")


@contextmanager
def server_manager(server_queue: Queue[ServerInstanceInfo], start_servers: bool):
    global PROCS
    servers_info = run_servers(
        num_inst=GeneralConfig.SERVER_COUNT,
        start_port=ServerConfig.VSHARP_INSTANCES_START_PORT,
        start_servers=start_servers,
    )

    for server_info in servers_info:
        server_queue.put(server_info)
    try:
        yield
    finally:
        for proc in PROCS:
            kill_server(proc, forget=False)
        PROCS = []


def main():
    global SERVER_INSTANCES, PROCS, RESULTS, START_SERVERS
    parser = argparse.ArgumentParser(description="V# instances launcher")
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="dont launch servers if set",
    )
    args = parser.parse_args()
    START_SERVERS = not args.debug

    # Queue[ServerInstanceInfo]
    SERVER_INSTANCES = Queue()
    PROCS = []
    RESULTS = []

    with server_manager(SERVER_INSTANCES, start_servers=START_SERVERS):
        app = web.Application()
        app.add_routes(routes)
        web.run_app(app, port=BrokerConfig.BROKER_PORT)


if __name__ == "__main__":
    main()
