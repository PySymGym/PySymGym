"""Unit tests for the result models in ``common.classes``."""

import pytest

from common.classes import GameFailed, GameResult

pytestmark = pytest.mark.unit


def test_game_result_str_handles_missing_coverage() -> None:
    result = GameResult(steps_count=1, tests_count=2, errors_count=3)
    assert str(result) == "(None, 2, 1, 3)"


def test_game_failed_str_is_constant_and_keeps_reason() -> None:
    failed = GameFailed("no path")
    assert str(failed) == "FAILED"
    assert failed.reason == "no path"
