import subprocess

import func_timeout

from src.psstrategy import AIStrategy, BasePSStrategy
from src.structs import LaunchInfo


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

    def runner_fun(call, wdir):
        return subprocess.check_output(call, stderr=subprocess.STDOUT, cwd=wdir).decode(
            "utf-8"
        )

    try:
        runner_output = func_timeout.func_timeout(
            timeout=timeout + 1, func=runner_fun, kwargs={"call": call, "wdir": wdir}
        )

    except subprocess.CalledProcessError:
        raise
    except func_timeout.FunctionTimedOut:
        raise

    return " ".join(map(str, call)), runner_output
