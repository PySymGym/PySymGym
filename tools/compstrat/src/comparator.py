import enum
import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from attrs import define


@define
class Color:
    name: str
    r: int
    g: int
    b: int
    a: float = 1

    def to_rgba(self):
        return (self.r / 255, self.g / 255, self.b / 255, self.a)

    @staticmethod
    def from_hex(hex_str: str):
        hex_str = hex_str.lstrip("#")
        return Color(
            r=int(hex_str[0:2], 16),
            g=int(hex_str[2:4], 16),
            b=int(hex_str[4:6], 16),
            name=hex_str,
        )


@define
class Strategy:
    name: str
    df: pd.DataFrame
    color: Color
    ecolor: Color


class DataSourceType(enum.Enum):
    # Outer join dataframe
    OUTER_JOIN_DF = "OUTER_JOIN_DF"
    # Inner join dataframe
    INNER_JOIN_DF = "INNER_JOIN_DF"
    # Inner join dataframe with equal coverage
    INNER_JOIN_COVERAGE_EQ_DF = "INNER_JOIN_COVERAGE_EQ_DF"


@define
class CompareConfig:
    datasource: DataSourceType
    by_column: str
    metric: str
    divider_line: bool = False
    less_is_winning: bool = False
    exp_name: str = None
    scale: str = None


class Comparator:
    def __init__(
        self,
        strat1: Strategy,
        strat2: Strategy,
        savedir: str,
        eq_color: Color = Color("black", 0, 0, 0, 1),
        eq_ecolor: Color = Color("translucent_black", 0, 0, 0, 0.5),
    ) -> None:
        self.savedir = savedir
        self.strat1 = strat1
        self.strat2 = strat2
        self.result_count_df = pd.DataFrame(
            columns=[f"{strat1.name}_won", f"{strat2.name}_won", "eq"]
        )
        self.eq_color = eq_color
        self.eq_ecolor = eq_ecolor

        with open(os.path.join(self.savedir, "symdiff_starts_methods.json"), "w") as f:
            json.dump(
                list(set(strat1.df.index).symmetric_difference(set(strat2.df.index))),
                f,
                indent=4,
            )

        self.inner_df = self.strat1.df.merge(
            self.strat2.df,
            on="method",
            how="inner",
            suffixes=(f"_{self.strat1.name}", f"_{self.strat2.name}"),
        )

        self.inner_df.mean().to_csv(os.path.join(self.savedir, "mean_statistics.csv"))

        self.inner_coverage_eq = self.inner_df.loc[
            self.inner_df[f"coverage_{self.strat1.name}"]
            == self.inner_df[f"coverage_{self.strat2.name}"]
        ]

    def _drop_failed(self, df: pd.DataFrame) -> pd.DataFrame:
        failed = df[(df["coverage"] == -1)].index
        return df.drop(failed)

    def drop_failed(self) -> int:
        self.strat1.df = self._drop_failed(self.strat1.df)
        self.strat2.df = self._drop_failed(self.strat2.df)

    def _compare(self, config: CompareConfig):
        def left_win_comparison(left, right):
            if config.less_is_winning:
                return left < right
            return left > right

        strat1_col, strat2_col = (
            f"{config.by_column}_{self.strat1.name}",
            f"{config.by_column}_{self.strat2.name}",
        )
        strat1_min_col, strat2_min_col = (
            f"{config.by_column}_min_{self.strat1.name}",
            f"{config.by_column}_min_{self.strat2.name}",
        )
        strat1_max_col, strat2_max_col = (
            f"{config.by_column}_max_{self.strat1.name}",
            f"{config.by_column}_max_{self.strat2.name}",
        )
        comparison_datasource_description = "datasource: "
        match config.datasource:
            case DataSourceType.OUTER_JOIN_DF:
                all_results = self.inner_df
                comparison_datasource_description += (
                    "methods, completed by both strats (inner join)"
                )
            case DataSourceType.INNER_JOIN_DF:
                all_results = self.inner_df
                comparison_datasource_description += (
                    "methods, completed by both strats (inner join)"
                )
            case DataSourceType.INNER_JOIN_COVERAGE_EQ_DF:
                all_results = self.inner_coverage_eq
                comparison_datasource_description += "methods, completed by both strats with equal coverage (inner join, cov1 == cov2)"
        strat1_win = all_results.loc[
            left_win_comparison(
                all_results[strat1_col],
                all_results[strat2_col],
            )
        ]
        strat2_win = all_results.loc[
            left_win_comparison(
                all_results[strat2_col],
                all_results[strat1_col],
            )
        ]
        eq = all_results.loc[all_results[strat1_col] == all_results[strat2_col]]

        if config.scale:
            plt.xscale(config.scale)
            plt.yscale(config.scale)

        if config.divider_line:
            plt.axline([0, 0], [1, 1])

        # check if exp_name already exists in df:
        if config.exp_name in self.result_count_df.index:
            warnings.warn(f"Overwriting {config.exp_name} in result_count_df")

        self.result_count_df.loc[config.exp_name] = [
            len(strat1_win),
            len(strat2_win),
            len(eq),
        ]
        plt.errorbar(
            x=strat1_win[strat2_col],
            y=strat1_win[strat1_col],
            xerr=[
                strat1_win[strat2_col] - strat1_win[strat2_min_col],
                strat1_win[strat2_max_col] - strat1_win[strat2_col],
            ],
            yerr=[
                strat1_win[strat1_col] - strat1_win[strat1_min_col],
                strat1_win[strat1_max_col] - strat1_win[strat1_col],
            ],
            ecolor=self.strat1.ecolor.to_rgba(),
            ls="none",
            capsize=3,
            marker="o",
            color=self.strat1.color.to_rgba(),
        )
        plt.errorbar(
            x=strat2_win[strat2_col],
            y=strat2_win[strat1_col],
            xerr=[
                strat2_win[strat2_col] - strat2_win[strat2_min_col],
                strat2_win[strat2_max_col] - strat2_win[strat2_col],
            ],
            yerr=[
                strat2_win[strat1_col] - strat2_win[strat1_min_col],
                strat2_win[strat1_max_col] - strat2_win[strat1_col],
            ],
            ecolor=self.strat2.ecolor.to_rgba(),
            ls="none",
            capsize=3,
            marker="o",
            color=self.strat2.color.to_rgba(),
        )
        plt.errorbar(
            x=eq[strat2_col],
            y=eq[strat1_col],
            xerr=[
                eq[strat2_col] - eq[strat2_min_col],
                eq[strat2_max_col] - eq[strat2_col],
            ],
            yerr=[
                eq[strat1_col] - eq[strat1_min_col],
                eq[strat1_max_col] - eq[strat1_col],
            ],
            ecolor=self.eq_ecolor.to_rgba(),
            ls="none",
            capsize=3,
            marker="o",
            color=self.eq_color.to_rgba(),
        )

        plt.xlabel(
            f"{self.strat2.name} {config.by_column}, {config.metric}\n\n"
            f"{config.by_column} {comparison_datasource_description}, {config.scale}\n"
            f"{self.strat1.name} ({self.strat1.color.name}) won: {len(strat1_win)}, "
            f"{self.strat2.name} ({self.strat2.color.name}) won: {len(strat2_win)}, \n eq ({self.eq_color.name}): {len(eq)}"
        )
        plt.ylabel(f"{self.strat1.name} {config.by_column}, {config.metric}")

        savename = (
            f"{config.on}.pdf" if config.exp_name is None else f"{config.exp_name}.pdf"
        )
        plt.savefig(
            os.path.join(self.savedir, savename), format="pdf", bbox_inches="tight"
        )
        plt.clf()

    def compare(self, configs: list[CompareConfig]):
        for config in tqdm.tqdm(
            configs, desc=f"{self.strat1.name} vs {self.strat2.name}"
        ):
            self._compare(config)

        self.result_count_df.to_csv(os.path.join(self.savedir, "result_count.csv"))
