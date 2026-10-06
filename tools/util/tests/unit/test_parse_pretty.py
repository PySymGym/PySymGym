"""Unit tests for the ``parse_pretty`` results-table parser."""

import pytest
from parse_pretty import parse_pretty, parse_stats

pytestmark = pytest.mark.unit

PRETTY_TABLE = [
    "|               | A_Method_0 | B_Method_0 |\n",
    "| Common Model  | actual %: 88.50, steps: 12, test count: 5 error count: 2 "
    "| actual %: 90.00, steps: 10, test count: 4 error count: 1 |\n",
]


def test_parse_stats_extracts_all_fields() -> None:
    stats = "actual %: 88.50, steps: 12, test count: 5 error count: 2"
    assert parse_stats(stats) == (12, 5, 2, 88.5)


def test_parse_stats_raises_on_malformed_input() -> None:
    with pytest.raises(AttributeError):
        parse_stats("no statistics here")


def test_parse_pretty_builds_one_row_per_method() -> None:
    df = parse_pretty(PRETTY_TABLE)

    assert df["method"].tolist() == ["A", "B"]
    assert df["steps"].tolist() == [12, 10]
    assert df["tests"].tolist() == [5, 4]
    assert df["errors"].tolist() == [2, 1]
    assert df["coverage"].tolist() == [88.5, 90.0]
