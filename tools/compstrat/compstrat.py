import argparse
import os
import pathlib
import typing as t

import attrs
import pandas as pd

from src.comparator import Color, Comparator, CompareConfig, Strategy
from src.compstrat_config_extractor import read_configs


@attrs.define
class Args:
    strat1: str
    run1: str
    strat2: str
    run2: str
    savedir: pathlib.Path
    configs: t.Sequence[CompareConfig]


def entrypoint(args: Args):
    def create(strat, run, color):
        return Strategy(strat, pd.read_csv(run), color)

    philippine_orange = Color(255, 115, 0, "orange")
    blue_sparkle = Color(0, 119, 255, "blue")
    os.makedirs(args.savedir, exist_ok=True)
    comparator = Comparator(
        strat1=create(args.strat1, args.run1, philippine_orange),
        strat2=create(args.strat2, args.run2, blue_sparkle),
        savedir=args.savedir,
    )
    comparator.compare(args.configs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s1",
        "--strat1",
        type=str,
        required=True,
        help="Name of the first strategy",
    )
    parser.add_argument(
        "-r1",
        "--run1",
        type=str,
        required=True,
        help="Path to the first strategy run result",
    )
    parser.add_argument(
        "-s2",
        "--strat2",
        type=str,
        required=True,
        help="Name of the second strategy",
    )
    parser.add_argument(
        "-r2",
        "--run2",
        type=str,
        required=True,
        help="Path to ther second strategy run result",
    )
    parser.add_argument(
        "-cp",
        "--configs-path",
        type=pathlib.Path,
        required=True,
        help="Path to ther second strategy run result",
    )
    parser.add_argument(
        "--savedir",
        type=pathlib.Path,
        required=False,
        default="report",
        help="Path to save results to",
    )
    args = parser.parse_args()

    entrypoint(
        Args(
            strat1=args.strat1,
            run1=args.run1,
            strat2=args.strat2,
            run2=args.run2,
            savedir=args.savedir,
            configs=read_configs(args.configs_path),
        )
    )


if __name__ == "__main__":
    main()
