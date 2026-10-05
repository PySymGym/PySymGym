"""Unit tests for the compstrat color helper and failure filtering."""

import pandas as pd
import pytest
from src.comparator import Color, Comparator

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
