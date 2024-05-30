import argparse
import itertools
import logging
import pathlib
import subprocess
from datetime import datetime

import attrs
import func_timeout
import pandas as pd
import tqdm

from src.parsing import load_config, parse_runner_output
from src.psstrategy import BasePSStrategy
from src.structs import RunResult
from src.subprocess_calls import call_test_runner

timestamp = datetime.fromtimestamp(datetime.now().timestamp())


def setup_logging(strategy_name: str):
    logging.basicConfig(
        filename=f"{strategy_name}_{timestamp}.log",
        format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
        level=logging.INFO,
    )


AssemblyInfo = tuple[pathlib.Path, pathlib.Path]


@attrs.define
class Args:
    strategy: BasePSStrategy
    timeout: int
    pysymgym_path: pathlib.Path
    assembly_infos: list[tuple[pathlib.Path, pathlib.Path]]


def entrypoint(args: Args) -> pd.DataFrame:
    runner_path = pathlib.Path(
        args.pysymgym_path
        / "GameServers/VSharp/VSharp.Runner/bin/Release/net7.0/VSharp.Runner.dll"
    ).resolve()

    setup_logging(strategy_name=args.strategy.name)

    assembled = list(
        itertools.chain(
            *[
                load_config(
                    pathlib.Path(dll_path).resolve(),
                    pathlib.Path(launch_info).resolve(),
                )
                for dll_path, launch_info in args.assembly_infos
            ]
        )
    )

    logging.info(args)
    results = []

    for launch_info in tqdm.tqdm(assembled, desc=args.strategy.name):
        try:
            call, runner_output = call_test_runner(
                path_to_runner=runner_path,
                launch_info=launch_info,
                strategy=args.strategy,
                wdir=runner_path.parent,
                timeout=args.timeout,
            )
        except subprocess.CalledProcessError as cpe:
            logging.error(
                f"""
                runner threw {cpe} on {launch_info.method}, this method will be skipped
                runner output: {cpe.output}
                cmd: {cpe.cmd}
                """
            )
            continue
        except func_timeout.FunctionTimedOut as fto:
            logging.error(
                f"""
                runner timed out on {launch_info.method}, this method will be skipped
                cmd: {" ".join(map(str,fto.timedOutKwargs["call"]))}
                """
            )
            continue

        try:
            (
                total_time,
                test_generated,
                errs_generated,
                runner_coverage,
            ) = parse_runner_output(runner_output)
        except AttributeError as e:
            logging.error(
                f"""
                {e} thrown on {launch_info.method}, this method will be skipped
                runner output: {runner_output}
                cmd: {call}
                """
            )
            continue

        run_result = RunResult(
            method=launch_info.method,
            tests=test_generated,
            errors=errs_generated,
            coverage=runner_coverage,
            total_time_sec=total_time,
        )

        logging.info(f"method {launch_info.method} completed with {run_result}")

        results.append(attrs.asdict(run_result))

    df = pd.DataFrame(results)
    df.to_csv(f"{args.strategy}_{timestamp}.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--strategy", type=str, required=True, help="V# searcher strategy"
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=pathlib.Path,
        dest="model_path",
        required=False,
        help="Absolute path to AI model if AI strategy is selected",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, required=True, help="V# runner timeout"
    )
    parser.add_argument(
        "-ps",
        "--pysymgym-path",
        type=pathlib.Path,
        dest="pysymgym_path",
        help="Absolute path to PySymGym",
    )
    parser.add_argument(
        "-as",
        "--assembly-infos",
        type=pathlib.Path,
        dest="assembly_infos",
        action="append",
        nargs=2,
        metavar=("dlls-path", "launch-info-path"),
        help="Provide tuples: dir with dlls/assembly info file",
    )

    args = parser.parse_args()

    default_model_path = pathlib.Path(
        args.pysymgym_path / "GameServers/VSharp/VSharp.Explorer/models/model.onnx"
    ).resolve()
    entrypoint(
        Args(
            strategy=BasePSStrategy.parse(
                args.strategy, model_path=args.model_path or default_model_path
            ),
            timeout=args.timeout,
            pysymgym_path=args.pysymgym_path,
            assembly_infos=args.assembly_infos,
        )
    )


if __name__ == "__main__":
    main()
