"""Unit tests for ``ml.training.early_stopping``."""

import pytest
from common.config.optuna_config import OptimizationDirection
from ml.training.early_stopping import EarlyStopping

pytestmark = pytest.mark.unit


def test_warmup_always_continues_and_fills_the_window():
    early_stopping = EarlyStopping(state_len=3, tolerance=0.01)

    assert early_stopping.is_continue(10.0) is True
    assert early_stopping.is_continue(10.0) is True
    assert early_stopping.is_continue(10.0) is True
    assert early_stopping._state == [10.0, 10.0, 10.0]


def test_minimize_stops_when_the_value_no_longer_improves():
    early_stopping = EarlyStopping(state_len=2, tolerance=1.0)
    early_stopping.is_continue(10.0)
    early_stopping.is_continue(10.0)

    assert early_stopping.is_continue(10.5) is False
    assert early_stopping._state == [10.0, 10.0]


def test_minimize_continues_and_evicts_the_oldest_sample():
    early_stopping = EarlyStopping(state_len=2, tolerance=1.0)
    early_stopping.is_continue(10.0)
    early_stopping.is_continue(10.0)

    assert early_stopping.is_continue(5.0) is True
    assert early_stopping._state == [10.0, 5.0]


def test_minimize_continues_when_difference_equals_tolerance():
    early_stopping = EarlyStopping(state_len=1, tolerance=1.0)
    early_stopping.is_continue(10.0)

    assert early_stopping.is_continue(9.0) is True


def test_maximize_stops_when_the_value_no_longer_improves():
    early_stopping = EarlyStopping(
        state_len=2, tolerance=1.0, direction=OptimizationDirection.MAXIMIZE
    )
    early_stopping.is_continue(10.0)
    early_stopping.is_continue(10.0)

    assert early_stopping.is_continue(9.0) is False
    assert early_stopping.is_continue(12.0) is True
