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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#592: a pretty table with more than one model row raises "
        "ValueError ('too many values to unpack') at the transpose; "
        "#592's fix spec pins the N-model-row transpose but not the "
        "post-fix output layout, so this pins the crash-level property"
    ),
)
def test_parse_pretty_handles_multiple_model_rows() -> None:
    df = parse_pretty(TWO_MODEL_TABLE)

    assert not df.empty
