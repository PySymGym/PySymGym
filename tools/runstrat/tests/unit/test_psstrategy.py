"""Unit tests for the runstrat path-selection strategies."""

from pathlib import Path

import pytest
from src.psstrategy import (
    AIStrategy,
    BasePSStrategy,
    ExecutionTreeContributedCoverageStrategy,
)

pytestmark = pytest.mark.unit


def test_parse_ai_strategy_with_model_path() -> None:
    strategy = BasePSStrategy.parse("AI", model_path=Path("model.onnx"))
    assert isinstance(strategy, AIStrategy)
    assert strategy.model_path == Path("model.onnx")


def test_parse_execution_tree_strategy() -> None:
    strategy = BasePSStrategy.parse("ExecutionTreeContributedCoverage")
    assert isinstance(strategy, ExecutionTreeContributedCoverageStrategy)


def test_parse_ai_without_model_path_asserts() -> None:
    with pytest.raises(AssertionError):
        BasePSStrategy.parse("AI")


def test_parse_unknown_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        BasePSStrategy.parse("unknown")
