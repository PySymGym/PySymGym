"""Unit tests for the compstrat color helper and failure filtering."""

from pathlib import Path

import pandas as pd
import pytest
from compstrat.comparator import (
    Color,
    Comparator,
    CompareConfig,
    DataSourceType,
    Strategy,
)

pytestmark = pytest.mark.unit


def test_color_from_hex_normalizes_to_rgba() -> None:
    color = Color.from_hex("#FF7300")
    assert color.to_rgba() == (1.0, 115 / 255, 0.0, 1)


def test_color_keeps_default_alpha() -> None:
    assert Color("orange", 255, 115, 0).to_rgba() == (1.0, 115 / 255, 0.0, 1)


def test_drop_failed_removes_coverage_minus_one() -> None:
    df = pd.DataFrame({"coverage": [1, -1, 2]}, index=["a", "b", "c"])

    result = Comparator._drop_failed(object(), df)

    assert list(result.index) == ["a", "c"]


def _coverage_strategy(name: str, index: list[str], coverage: list[float]) -> Strategy:
    color = Color(name, 1, 2, 3)
    return Strategy(
        name=name,
        df=pd.DataFrame(
            {
                "coverage": coverage,
                "coverage_min": coverage,
                "coverage_max": coverage,
            },
            index=pd.Index(index, name="method"),
        ),
        color=color,
        ecolor=color,
    )


def test_outer_join_datasource_covers_union_and_counts_comparable_rows(
    tmp_path: Path,
) -> None:
    comparator = Comparator(
        strat1=_coverage_strategy("A", ["m1", "m2", "m3"], [1.0, 2.0, 3.0]),
        strat2=_coverage_strategy("B", ["m2", "m3", "m4"], [5.0, 3.0, 2.0]),
        savedir=str(tmp_path),
    )

    assert list(comparator.inner_df.index) == ["m2", "m3"]
    assert list(comparator.outer_df.index) == ["m1", "m2", "m3", "m4"]

    # m1 (A only) and m4 (B only) carry NaN on the missing side, so they
    # cannot win, lose, or tie: the outer-join counts equal the
    # hand-computed intersection counts (m2: B wins, m3: tie).
    comparator._compare(
        CompareConfig(
            datasource=DataSourceType.OUTER_JOIN_DF,
            by_column="coverage",
            metric="%",
            exp_name="coverage_outer",
        )
    )
    outer_row = comparator.result_count_df.loc["coverage_outer"]
    assert [outer_row["A_won"], outer_row["B_won"], outer_row["eq"]] == [0, 1, 1]

    comparator._compare(
        CompareConfig(
            datasource=DataSourceType.INNER_JOIN_DF,
            by_column="coverage",
            metric="%",
            exp_name="coverage_inner",
        )
    )
    inner_row = comparator.result_count_df.loc["coverage_inner"]
    assert [inner_row["A_won"], inner_row["B_won"], inner_row["eq"]] == [0, 1, 1]
