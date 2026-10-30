import enum
import json
import os
from typing import Optional
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from attrs import define

from compstrat.preprocessing import MIN_POSTFIX, MAX_POSTFIX


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
    exp_name: Optional[str] = None
    scale: Optional[str] = None
    # Pipeline-level flag: failed runs (coverage == -1) are excluded from
    # every comparison unless any config sets this to false (see
    # tools/README.md).
    drop_failed: bool = True


class Comparator:
    def __init__(
        self,
        strat1: Strategy,
        strat2: Strategy,
        savedir: str,
        eq_color: Color | None = None,
        eq_ecolor: Color | None = None,
    ) -> None:
        if eq_color is None:
            eq_color = Color("black", 0, 0, 0, 1)
        if eq_ecolor is None:
            eq_ecolor = Color("translucent_black", 0, 0, 0, 0.5)
        self.savedir = savedir
        self.strat1 = strat1
        self.strat2 = strat2
        self.result_count_df = pd.DataFrame(
            columns=[f"{strat1.name}_won", f"{strat2.name}_won", "eq"]
        )
        self.eq_color = eq_color
        self.eq_ecolor = eq_ecolor

        self._build_joins()

    def _build_joins(self) -> None:
        """Compute the joined dataframes and the derived statistics.

        Joins the strategy dataframes on the shared ``method`` index level
        (inner and outer) and writes the per-join artifacts:
        ``symdiff_starts_methods.json`` (methods present in only one
        strategy), ``mean_statistics.csv`` (inner-join column means), and
        the equal-coverage subset of the inner join. Called from
        ``__init__`` and again by ``drop_failed`` after the strategy
        dataframes change.
        """
        with open(os.path.join(self.savedir, "symdiff_starts_methods.json"), "w") as f:
            json.dump(
                list(
                    set(self.strat1.df.index).symmetric_difference(
                        set(self.strat2.df.index)
                    )
                ),
                f,
                indent=4,
            )

        suffixes = (f"_{self.strat1.name}", f"_{self.strat2.name}")
        self.inner_df = self.strat1.df.merge(
            self.strat2.df,
            on="method",
            how="inner",
            suffixes=suffixes,
        )
        self.outer_df = self.strat1.df.merge(
            self.strat2.df,
            on="method",
            how="outer",
            suffixes=suffixes,
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
        """Drop failed runs (``coverage == -1``) from both strategies.

        Rebuilds the joined dataframes and the derived statistics
        (``_build_joins``) so every downstream comparison sees the
        filtered data.

        Returns:
            The total number of dropped rows across both strategies.
        """
        kept1 = self._drop_failed(self.strat1.df)
        kept2 = self._drop_failed(self.strat2.df)
        dropped = (len(self.strat1.df) - len(kept1)) + (
            len(self.strat2.df) - len(kept2)
        )
        self.strat1.df = kept1
        self.strat2.df = kept2
        self._build_joins()
        return dropped

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
            f"{config.by_column}{MIN_POSTFIX}_{self.strat1.name}",
            f"{config.by_column}{MIN_POSTFIX}_{self.strat2.name}",
        )
        strat1_max_col, strat2_max_col = (
            f"{config.by_column}{MAX_POSTFIX}_{self.strat1.name}",
            f"{config.by_column}{MAX_POSTFIX}_{self.strat2.name}",
        )
        comparison_datasource_description = "datasource: "
        match config.datasource:
            case DataSourceType.OUTER_JOIN_DF:
                all_results = self.outer_df
                comparison_datasource_description += (
                    "methods, completed by at least one strat (outer join)"
                )
            case DataSourceType.INNER_JOIN_DF:
                all_results = self.inner_df
                comparison_datasource_description += (
                    "methods, completed by both strats (inner join)"
                )
            case DataSourceType.INNER_JOIN_COVERAGE_EQ_DF:
                all_results = self.inner_coverage_eq
                comparison_datasource_description += "methods, completed by both strats with equal coverage (inner join, cov1 == cov2)"
        # For the outer join, methods completed by only one strategy carry
        # NaN on the missing side. Every comparison below evaluates to
        # False for NaN, so those methods enter none of the win/loss/eq
        # subsets and are absent from the plot; no NaN ever reaches the
        # statistics or matplotlib.
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
            warnings.warn(
                f"Overwriting {config.exp_name} in result_count_df", stacklevel=2
            )

        self.result_count_df.loc[config.exp_name] = [
            len(strat1_win),
            len(strat2_win),
            len(eq),
        ]
        # x/y as numpy arrays, err as lists of Python floats: matplotlib's
        # _upcast_err calls math.isfinite() on each err element, which warns
        # for single-element Series ("Calling float on a single element
        # Series is deprecated") and for single-element ndarrays ("Conversion
        # of an array with ndim > 0 to a scalar is deprecated"); plain lists
        # raise TypeError there instead, which matplotlib handles silently.
        plt.errorbar(
            x=strat1_win[strat2_col].to_numpy(),
            y=strat1_win[strat1_col].to_numpy(),
            xerr=[
                (strat1_win[strat2_col] - strat1_win[strat2_min_col]).tolist(),
                (strat1_win[strat2_max_col] - strat1_win[strat2_col]).tolist(),
            ],
            yerr=[
                (strat1_win[strat1_col] - strat1_win[strat1_min_col]).tolist(),
                (strat1_win[strat1_max_col] - strat1_win[strat1_col]).tolist(),
            ],
            ecolor=self.strat1.ecolor.to_rgba(),
            ls="none",
            capsize=3,
            marker="o",
            color=self.strat1.color.to_rgba(),
        )
        plt.errorbar(
            x=strat2_win[strat2_col].to_numpy(),
            y=strat2_win[strat1_col].to_numpy(),
            xerr=[
                (strat2_win[strat2_col] - strat2_win[strat2_min_col]).tolist(),
                (strat2_win[strat2_max_col] - strat2_win[strat2_col]).tolist(),
            ],
            yerr=[
                (strat2_win[strat1_col] - strat2_win[strat1_min_col]).tolist(),
                (strat2_win[strat1_max_col] - strat2_win[strat1_col]).tolist(),
            ],
            ecolor=self.strat2.ecolor.to_rgba(),
            ls="none",
            capsize=3,
            marker="o",
            color=self.strat2.color.to_rgba(),
        )
        plt.errorbar(
            x=eq[strat2_col].to_numpy(),
            y=eq[strat1_col].to_numpy(),
            xerr=[
                (eq[strat2_col] - eq[strat2_min_col]).tolist(),
                (eq[strat2_max_col] - eq[strat2_col]).tolist(),
            ],
            yerr=[
                (eq[strat1_col] - eq[strat1_min_col]).tolist(),
                (eq[strat1_max_col] - eq[strat1_col]).tolist(),
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
