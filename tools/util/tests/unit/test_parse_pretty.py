"""Unit tests for the ``parse_pretty`` results-table parser."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from parse_pretty import main, parse_pretty, parse_stats

pytestmark = pytest.mark.unit

PRETTY_TABLE = [
    "|               | A_Method_0 | B_Method_0 |\n",
    "| Common Model  | actual %: 88.50, steps: 12, test count: 5 error count: 2 "
    "| actual %: 90.00, steps: 10, test count: 4 error count: 1 |\n",
]


def _assert_pretty_df(df: pd.DataFrame) -> None:
    assert df["method"].tolist() == ["A", "B"]
    assert df["steps"].tolist() == [12, 10]
    assert df["tests"].tolist() == [5, 4]
    assert df["errors"].tolist() == [2, 1]
    assert df["coverage"].tolist() == [88.5, 90.0]


def test_parse_stats_extracts_all_fields() -> None:
    stats = "actual %: 88.50, steps: 12, test count: 5 error count: 2"
    assert parse_stats(stats) == (12, 5, 2, 88.5)


def test_parse_stats_raises_on_malformed_input() -> None:
    with pytest.raises(AttributeError):
        parse_stats("no statistics here")


def test_parse_stats_rejects_non_digit_counts() -> None:
    # The fixed \d+ regexes reject a non-digit where a count belongs (the
    # old "any char" regexes accepted "x12" as a count).
    with pytest.raises(AttributeError):
        parse_stats("actual %: 88.50, steps: x12, test count: 5 error count: 2")


def test_parse_pretty_builds_one_row_per_method() -> None:
    df = parse_pretty(PRETTY_TABLE)

    _assert_pretty_df(df)


def test_parse_pretty_skips_epoch_and_separator_lines() -> None:
    table = [
        "Epoch# 1\n",
        PRETTY_TABLE[0],
        "|---------------+------------+------------|\n",
        PRETTY_TABLE[1],
    ]

    df = parse_pretty(table)

    _assert_pretty_df(df)


def test_main_writes_parsed_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "table.txt"
    input_path.write_text("".join(PRETTY_TABLE))
    output_path = tmp_path / "parsed.csv"

    monkeypatch.setattr(
        sys, "argv", ["parse_pretty", "-in", str(input_path), "-o", str(output_path)]
    )
    main()

    df = pd.read_csv(output_path)
    assert df.columns.tolist() == ["method", "steps", "tests", "errors", "coverage"]
    _assert_pretty_df(df)


def test_main_default_output_is_parsed_csv_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "table.txt"
    input_path.write_text("".join(PRETTY_TABLE))

    monkeypatch.setattr(sys, "argv", ["parse_pretty", "-in", str(input_path)])
    monkeypatch.chdir(tmp_path)
    main()

    df = pd.read_csv(tmp_path / "parsed.csv")
    _assert_pretty_df(df)


TWO_MODEL_TABLE = [
    "|               | A_Method_0 | B_Method_0 |\n",
    "| Model One     | actual %: 88.50, steps: 12, test count: 5 error count: 2 "
    "| actual %: 90.00, steps: 10, test count: 4 error count: 1 |\n",
    "| Model Two     | actual %: 80.00, steps: 20, test count: 6 error count: 0 "
    "| actual %: 85.25, steps: 15, test count: 3 error count: 1 |\n",
]


def test_parse_pretty_handles_multiple_model_rows() -> None:
    # #592: the former strict xfail pinned the crash-level property (a
    # table with more than one model row raised ValueError at the
    # transpose). Post-fix layout, owned by #592: one row per
    # (model, method) pair, model-major, with a leading model column.
    df = parse_pretty(TWO_MODEL_TABLE)

    assert not df.empty
    assert df.columns.tolist() == [
        "model",
        "method",
        "steps",
        "tests",
        "errors",
        "coverage",
    ]
    assert df["model"].tolist() == ["Model One", "Model One", "Model Two", "Model Two"]
    assert df["method"].tolist() == ["A", "B", "A", "B"]
    assert df["steps"].tolist() == [12, 10, 20, 15]
    assert df["tests"].tolist() == [5, 4, 6, 3]
    assert df["errors"].tolist() == [2, 1, 0, 1]
    assert df["coverage"].tolist() == [88.5, 90.0, 80.0, 85.25]


def test_parse_pretty_mismatched_row_width_raises_value_error() -> None:
    table = [
        "|               | A_Method_0 | B_Method_0 |\n",
        "| Model One     | actual %: 88.50, steps: 12, test count: 5 error count: 2 |\n",
    ]

    with pytest.raises(
        ValueError, match="has 1 stats cells, but the header declares 2"
    ):
        parse_pretty(table)


def test_parse_pretty_empty_table_returns_empty_frame() -> None:
    df = parse_pretty([])

    assert df.empty
    assert df.columns.tolist() == ["method", "steps", "tests", "errors", "coverage"]
