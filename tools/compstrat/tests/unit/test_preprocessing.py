"""Unit tests for the compstrat preprocessing of strategy runs."""

import warnings

import pandas as pd
import pytest
from compstrat.preprocessing import _union_index, preprocess

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

    with pytest.warns(UserWarning, match="only in strat1: m1; only in strat2: m4"):
        strat1_df, strat2_df = preprocess(strat1_runs, strat2_runs)

    assert list(strat1_df.index) == ["m2", "m3"]
    assert list(strat2_df.index) == ["m2", "m3"]


def test_preprocess_does_not_warn_when_indices_match() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        preprocess([_run([1.0, 2.0], [10, 20])], [_run([5.0, 6.0], [50, 60])])


def test_preprocess_warning_lists_dropped_methods_bounded() -> None:
    strat1_runs = [
        pd.DataFrame({"coverage": [1.0] * 15}, index=[f"m{i}" for i in range(15)])
    ]
    strat2_runs = [pd.DataFrame({"coverage": [2.0]}, index=["m100"])]

    with pytest.warns(UserWarning, match=r"and 5 more") as record:
        preprocess(strat1_runs, strat2_runs)

    # The names are sorted lexicographically: the first ten of m0-m14 are
    # m0, m1, m10, m11, m12, m13, m14, m2, m3, m4; the rest are elided.
    message = str(record.list[0].message)
    assert "m0" in message
    assert "m4" in message
    assert "m5" not in message
    assert "m9" not in message
    assert "m100" in message


def test_preprocess_handles_empty_runs() -> None:
    empty = pd.DataFrame({"coverage": pd.Series(dtype=float)})

    strat1_df, strat2_df = preprocess([empty], [empty])

    assert list(strat1_df.columns) == ["coverage", "coverage_min", "coverage_max"]
    assert strat1_df.empty
    assert strat2_df.empty


def test_union_index_of_no_runs_is_empty() -> None:
    # Reachable via asymmetric input (preprocess([], [df])); preprocess
    # itself rejects two empty lists at the concat.
    assert _union_index([]).empty


def test_preprocess_single_run_has_equal_min_and_max() -> None:
    strat1_df, strat2_df = preprocess(
        [_run([1.0, 2.0], [10, 20])], [_run([5.0, 6.0], [50, 60])]
    )

    assert list(strat2_df["coverage"]) == [5.0, 6.0]
    assert list(strat2_df["coverage_min"]) == [5, 6]
    assert list(strat2_df["coverage_max"]) == [5, 6]
