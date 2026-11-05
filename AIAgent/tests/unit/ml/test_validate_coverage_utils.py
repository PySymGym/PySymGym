"""Unit tests for ``ml.validation.coverage.validate_coverage_utils``."""

import pytest
from ml.validation.coverage.validate_coverage_utils import catch_return_exception

pytestmark = pytest.mark.unit


def test_returns_the_value_on_success():
    @catch_return_exception
    def add(a, b=0):
        return a + b

    assert add(1, b=2) == 3


def test_preserves_the_wrapped_function_name():
    @catch_return_exception
    def some_function():
        return None

    assert some_function.__name__ == "some_function"


def test_returns_the_exception_instance_on_failure():
    error = ValueError("boom")

    @catch_return_exception
    def failing():
        raise error

    assert failing() is error


def test_passes_through_args_and_kwargs():
    @catch_return_exception
    def echo(*args, **kwargs):
        return args, kwargs

    assert echo(1, 2, x=3) == ((1, 2), {"x": 3})
