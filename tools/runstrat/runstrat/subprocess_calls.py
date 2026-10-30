import os
import signal
import subprocess

import func_timeout

from runstrat.psstrategy import AIStrategy, BasePSStrategy
from runstrat.structs import LaunchInfo


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the child's process group (it leads its own session).

    ``dotnet`` may spawn children of its own, which killing the direct
    child would orphan. A group that already exited is fine.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def call_test_runner(
    path_to_runner: str,
    launch_info: LaunchInfo,
    strategy: BasePSStrategy,
    wdir: str,
    timeout: int,
):
    call = [
        "dotnet",
        path_to_runner,
        "--method",
        launch_info.method,
        launch_info.dll,
        "--timeout",
        str(timeout),
        "--strat",
        strategy.name,
        "--check-coverage",
    ]

    match strategy:
        case AIStrategy(_, model_path):
            call += (
                "--model",
                model_path,
            )

    # The child leads its own session so the whole group (dotnet may spawn
    # children) can be killed on timeout. Waiting on the process directly
    # instead of via func_timeout, which would interrupt only the waiting
    # thread and orphan the runner.
    proc = subprocess.Popen(
        call,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=wdir,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout + 1)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.wait()
        raise func_timeout.FunctionTimedOut(
            timedOutAfter=timeout + 1,
            timedOutKwargs={"call": call, "wdir": wdir},
        ) from None

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, call, output=out)

    return " ".join(map(str, call)), out.decode("utf-8")
