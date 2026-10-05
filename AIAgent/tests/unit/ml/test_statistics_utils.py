"""Unit tests for ``ml.validation.statistics_utils``."""

from collections import namedtuple

import pytest
from ml.validation.statistics_utils import avg_by_attr

pytestmark = pytest.mark.unit

Result = namedtuple("Result", ["coverage_percent", "errors_number"])


def test_empty_results_return_minus_one():
    assert avg_by_attr([], "coverage_percent") == -1


def test_averages_the_named_attribute():
    results = [Result(10, 1), Result(20, 3), Result(30, 5)]

    assert avg_by_attr(results, "coverage_percent") == 20.0
    assert avg_by_attr(results, "errors_number") == 3.0


def test_single_result_is_its_own_average():
    assert avg_by_attr([Result(42, 7)], "coverage_percent") == 42.0


def test_missing_attribute_raises():
    with pytest.raises(AttributeError):
        avg_by_attr([Result(10, 1)], "does_not_exist")
