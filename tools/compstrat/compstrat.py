import argparse
import os
import pathlib
import typing as t

import pandas as pd
from attrs import define
from src.comparator import Color, Comparator, CompareConfig, Strategy
from src.config_extractor import read_configs
from src.preprocessing import preprocess


@define
class Args:
    strat1: str
    runs1: list[str]
    strat2: str
    runs2: list[str]
    savedir: pathlib.Path
    configs: t.Sequence[CompareConfig]


def entrypoint(args: Args):
    PHILIPPINE_ORANGE, TRANSLUCENT_PHILIPPINE_ORANGE = (
        Color("orange", 255, 115, 0),
        Color("translucent_orange", 255, 115, 0, 0.5),
    )
    BLUE_SPARKLE, TRANSLUCENT_BLUE_SPARKLE = (
        Color("blue", 0, 119, 255),
        Color("translucent_blue", 0, 119, 255, 0.5),
    )
    os.makedirs(args.savedir, exist_ok=True)
    strat1_df, strat2_df = preprocess(
        [pd.read_csv(run, index_col="method") for run in args.runs1],
        [pd.read_csv(run, index_col="method") for run in args.runs2],
    )
    comparator = Comparator(
        strat1=Strategy(
            name=args.strat1,
            df=strat1_df,
            color=PHILIPPINE_ORANGE,
            ecolor=TRANSLUCENT_PHILIPPINE_ORANGE,
        ),
        strat2=Strategy(
            name=args.strat2,
            df=strat2_df,
            color=BLUE_SPARKLE,
            ecolor=TRANSLUCENT_BLUE_SPARKLE,
        ),
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
        "--runs1",
        action="extend",
        nargs="+",
        type=str,
        required=True,
        help="Paths to the first strategy runs results",
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
        "--runs2",
        action="extend",
        nargs="+",
        type=str,
        required=True,
        help="Paths to the second strategy runs results",
    )
    parser.add_argument(
        "-cp",
        "--configs-path",
        type=pathlib.Path,
        required=True,
        help="Path to the compare configurations",
    )
    parser.add_argument(
        "--savedir",
        type=pathlib.Path,
        required=False,
        default="report",
        help="Path for saving the comparison results",
    )
    args = parser.parse_args()

    entrypoint(
        Args(
            strat1=args.strat1,
            runs1=args.runs1,
            strat2=args.strat2,
            runs2=args.runs2,
            savedir=args.savedir,
            configs=read_configs(args.configs_path),
        )
    )


if __name__ == "__main__":
    main()
