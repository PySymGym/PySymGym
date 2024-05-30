import enum
import json
import os
import warnings

import attrs
import matplotlib.pyplot as plt
import pandas as pd
import tqdm


@attrs.define
class Color:
    r: int
    g: int
    b: int

    name: str

    def to_rgb(self):
        return (self.r / 255, self.g / 255, self.b / 255)

    @staticmethod
    def from_hex(hex_str: str):
        hex_str = hex_str.lstrip("#")
        return Color(
            r=int(hex_str[0:2], 16),
            g=int(hex_str[2:4], 16),
            b=int(hex_str[4:6], 16),
            name=hex_str,
        )


@attrs.define
class Strategy:
    name: str
    df: pd.DataFrame
    color: Color


class DataSourceType(enum.Enum):
    # Outer join dataframe
    OUTER_JOIN = "OUTER_JOIN"
    # Inner join dataframe
    INNER_JOIN_DF = "INNER_JOIN_DF"
    # Inner join dataframe with equal coverage
    INNER_JOIN_COVERAGE_EQ_DF = "INNER_JOIN_COVERAGE_EQ_DF"


@attrs.define
class CompareConfig:
    datasource: DataSourceType
    by_column: str
    metric: str
    divider_line: bool = False
    less_is_winning: bool = False
    logscale: bool = False
    exp_name: str = None


class Comparator:
    def __init__(
        self,
        strat1: Strategy,
        strat2: Strategy,
        savedir: str,
        eq_color: Color = Color(0, 0, 0, "black"),
    ) -> None:
        self.savedir = savedir
        self.strat1 = strat1
        self.strat2 = strat2
        self.result_count_df = pd.DataFrame(
            columns=[f"{strat1.name}_won", f"{strat2.name}_won", "eq"]
        )
        self.eq_color = eq_color

        with open(os.path.join(self.savedir, "symdiff_starts_methods.json"), "w") as f:
            json.dump(
                list(
                    set(strat1.df["method"]).symmetric_difference(
                        set(strat2.df["method"])
                    )
                ),
                f,
                indent=4,
            )
        self.drop_failed()

        self.outer_df = self.strat1.df.merge(
            self.strat2.df,
            on="method",
            how="outer",
            suffixes=(self.strat1.name, self.strat2.name),
        )

        self.inner_df = self.strat1.df.merge(
            self.strat2.df,
            on="method",
            how="inner",
            suffixes=(self.strat1.name, self.strat2.name),
        )

        self.inner_coverage_eq = self.inner_df.loc[
            self.inner_df[f"coverage{self.strat1.name}"]
            == self.inner_df[f"coverage{self.strat2.name}"]
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

        match config.datasource:
            case DataSourceType.OUTER_JOIN:
                dataframe = self.inner_df
            case DataSourceType.INNER_JOIN_DF:
                dataframe = self.inner_df
            case DataSourceType.INNER_JOIN_COVERAGE_EQ_DF:
                dataframe = self.inner_coverage_eq

        strat1_win = dataframe.loc[
            left_win_comparison(
                dataframe[f"{config.by_column}{self.strat1.name}"],
                dataframe[f"{config.by_column}{self.strat2.name}"],
            )
        ]
        strat2_win = dataframe.loc[
            left_win_comparison(
                dataframe[f"{config.by_column}{self.strat2.name}"],
                dataframe[f"{config.by_column}{self.strat1.name}"],
            )
        ]
        eq = dataframe.loc[
            dataframe[f"{config.by_column}{self.strat1.name}"]
            == dataframe[f"{config.by_column}{self.strat2.name}"]
        ]

        scale = "linscale"
        if config.logscale:
            plt.xscale("log")
            plt.yscale("log")
            scale = "logscale"

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

        plt.scatter(
            strat1_win[f"{config.by_column}{self.strat2.name}"],
            strat1_win[f"{config.by_column}{self.strat1.name}"],
            color=[self.strat1.color.to_rgb()],
        )
        plt.scatter(
            strat2_win[f"{config.by_column}{self.strat2.name}"],
            strat2_win[f"{config.by_column}{self.strat1.name}"],
            color=[self.strat2.color.to_rgb()],
        )
        plt.scatter(
            eq[f"{config.by_column}{self.strat2.name}"],
            eq[f"{config.by_column}{self.strat1.name}"],
            color=self.eq_color.to_rgb(),
        )
        plt.xlabel(
            f"{self.strat2.name} {config.by_column}, {config.metric}\n\n"
            f"{config.by_column} comparison on the same methods, {scale}\n"
            f"{self.strat1.name} ({self.strat1.color.name}) won: {len(strat1_win)}, "
            f"{self.strat2.name} ({self.strat2.color.name}) won: {len(strat2_win)}, eq ({self.eq_color.name}): {len(eq)}"
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
