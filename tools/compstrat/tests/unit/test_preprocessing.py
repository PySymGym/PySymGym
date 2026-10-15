"""Unit tests for the compstrat preprocessing of strategy runs."""

import pandas as pd
import pytest
from src.preprocessing import preprocess

pytestmark = pytest.mark.unit


def _run(coverage: list[float], steps: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"coverage": coverage, "steps": steps}, index=["m1", "m2"])


def test_preprocess_averages_runs_and_tracks_min_max() -> None:
    strat1_runs = [_run([1.0, 2.0], [10, 20]), _run([3.0, 4.0], [30, 40])]
    strat2_runs = [_run([5.0, 6.0], [50, 60])]

    strat1_df, strat2_df = preprocess(strat1_runs, strat2_runs)

    assert list(strat1_df.columns) == [
        "coverage",
        "coverage_min",
        "coverage_max",
        "steps",
        "steps_min",
        "steps_max",
    ]
    assert list(strat1_df["coverage"]) == [2.0, 3.0]
    assert list(strat1_df["coverage_min"]) == [1, 2]
    assert list(strat1_df["coverage_max"]) == [3, 4]
    assert list(strat1_df["steps"]) == [20.0, 30.0]
    assert list(strat1_df["steps_min"]) == [10, 20]
    assert list(strat1_df["steps_max"]) == [30, 40]


def test_preprocess_uses_inner_join_index() -> None:
    strat1_runs = [
        pd.DataFrame({"coverage": [1.0, 2.0, 3.0]}, index=["m1", "m2", "m3"])
    ]
    strat2_runs = [
        pd.DataFrame({"coverage": [9.0, 8.0, 7.0]}, index=["m2", "m3", "m4"])
    ]

    strat1_df, strat2_df = preprocess(strat1_runs, strat2_runs)

    assert list(strat1_df.index) == ["m2", "m3"]
    assert list(strat2_df.index) == ["m2", "m3"]


def test_preprocess_handles_empty_runs() -> None:
    empty = pd.DataFrame({"coverage": pd.Series(dtype=float)})

    strat1_df, strat2_df = preprocess([empty], [empty])

    assert list(strat1_df.columns) == ["coverage", "coverage_min", "coverage_max"]
    assert strat1_df.empty
    assert strat2_df.empty


def test_preprocess_single_run_has_equal_min_and_max() -> None:
    strat1_df, strat2_df = preprocess(
        [_run([1.0, 2.0], [10, 20])], [_run([5.0, 6.0], [50, 60])]
    )

    assert list(strat2_df["coverage"]) == [5.0, 6.0]
    assert list(strat2_df["coverage_min"]) == [5, 6]
    assert list(strat2_df["coverage_max"]) == [5, 6]
