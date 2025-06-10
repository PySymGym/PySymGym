import argparse
import itertools
import logging
from pathlib import Path
import subprocess
from datetime import datetime
import os

import func_timeout
import pandas as pd
import tqdm
from attrs import asdict, define
from src.parsing import load_config, parse_runner_output
from src.psstrategy import BasePSStrategy
from src.structs import RunResult
from src.subprocess_calls import call_test_runner
from enum import Enum


DOTNET_VERSION = "7.0"
PATH_TO_VSHARP = Path(
    f"GameServers/VSharp/VSharp.Runner/bin/Release/net{DOTNET_VERSION}/VSharp.Runner.dll"
)
timestamp = datetime.fromtimestamp(datetime.now().timestamp())


def setup_logging(savedir: Path, strategy_name: str):
    logging.basicConfig(
        filename=os.path.join(savedir, f"{strategy_name}.log"),
        format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
        level=logging.INFO,
    )


AssemblyInfo = tuple[Path, Path]


class RunMode(Enum):
    BENCHMARKS = "Benchmarks"
    DEBUG = "Debug"


@define
class Args:
    strategy: BasePSStrategy
    timeout: int
    pysymgym_path: Path
    savedir: Path
    assembly_infos: list[AssemblyInfo]
    run_mode: RunMode = RunMode.BENCHMARKS


def entrypoint(args: Args) -> pd.DataFrame:
    runner_path = Path(args.pysymgym_path / PATH_TO_VSHARP).resolve()

    setup_logging(args.savedir, strategy_name=args.strategy.name)

    assembled = list(
        itertools.chain(
            *[
                load_config(
                    Path(dll_path).resolve(),
                    Path(launch_info).resolve(),
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
            if args.run_mode == RunMode.DEBUG:
                raise cpe
            continue
        except func_timeout.FunctionTimedOut as fto:
            logging.error(
                f"""
            runner timed out on {launch_info.method}, this method will be skipped
            cmd: {" ".join(map(str, fto.timedOutKwargs["call"]))}
            """
            )
            if args.run_mode == RunMode.DEBUG:
                raise fto
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
            if args.run_mode == RunMode.DEBUG:
                raise e
            continue

        run_result = RunResult(
            method=launch_info.method,
            tests=test_generated,
            errors=errs_generated,
            coverage=runner_coverage,
            total_time_sec=total_time,
        )

        logging.info(f"method {launch_info.method} completed with {run_result}")

        results.append(asdict(run_result))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.savedir, f"{args.strategy.name}.csv"), index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        required=True,
        help="V# searcher strategy. ExecutionTreeContributedCoverage and AI are supported.",
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=Path,
        required=False,
        help="Absolute path to AI model if AI strategy is selected",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, required=True, help="V# runner timeout (s)"
    )
    parser.add_argument(
        "-ps",
        "--pysymgym-path",
        type=Path,
        dest="pysymgym_path",
        required=False,
        help="Absolute path to PySymGym",
    )
    parser.add_argument(
        "-sd",
        "--savedir",
        type=Path,
        required=False,
        help="Path to save results to. Default is current directory.",
    )
    parser.add_argument(
        "-as",
        "--assembly-infos",
        type=Path,
        action="append",
        nargs=2,
        metavar=("dlls-path", "launch-info-path"),
        help="Provide tuples: dir with dlls/assembly info file",
    )

    args = parser.parse_args()

    default_savedir = args.savedir or Path(f"{timestamp}")
    os.makedirs(default_savedir)
    entrypoint(
        Args(
            strategy=BasePSStrategy.parse(args.strategy, model_path=args.model_path),
            timeout=args.timeout,
            pysymgym_path=args.pysymgym_path,
            savedir=default_savedir,
            assembly_infos=args.assembly_infos,
        )
    )


if __name__ == "__main__":
    main()
