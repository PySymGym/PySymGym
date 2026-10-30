"""Unit tests for the shared game-manager utilities.

``play_with_timeout`` is tested against the real ``func_timeout`` wrapper:
the inner-timeout case relies on the library's swallow-and-return-``None``
behaviour for a ``FunctionTimedOut`` raised inside the wrapped function
(#672).
"""

import pytest
from common.config.feature_config import FeatureConfig, SaveIfFailOrTimeout
from func_timeout import FunctionTimedOut
from ml.validation.coverage.game_managers.utils import (
    play_with_timeout,
    set_timeout_if_needed,
)

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def _feature_config(**overrides) -> FeatureConfig:
    return FeatureConfig(save_if_fail_or_timeout=SaveIfFailOrTimeout(**overrides))


def test_set_timeout_if_needed_wraps_only_when_enabled() -> None:
    func = lambda: 1  # noqa: E731

    assert set_timeout_if_needed(_feature_config(), func) is not func
    assert set_timeout_if_needed(_feature_config(enabled=False), func) is func


def test_play_with_timeout_passes_the_result_through() -> None:
    assert play_with_timeout(_feature_config(), lambda: (1, 2)) == (1, 2)


def test_play_with_timeout_raises_on_a_swallowed_inner_timeout() -> None:
    # The worker thread swallows the inner FunctionTimedOut and the wrapper
    # returns None; play_with_timeout translates that back into a raised
    # FunctionTimedOut carrying its own message (not the swallowed one).
    def inner():
        raise FunctionTimedOut(timedOutAfter=1800)

    with pytest.raises(FunctionTimedOut) as excinfo:
        play_with_timeout(_feature_config(), inner)

    assert "timed out internally" in str(excinfo.value)


def test_play_with_timeout_raises_on_a_none_return() -> None:
    # The game-play functions contractually return a non-None result, so a
    # None return under an active wrapper is always a timeout signal.
    with pytest.raises(FunctionTimedOut):
        play_with_timeout(_feature_config(), lambda: None)


def test_play_with_timeout_passes_through_when_disabled() -> None:
    # Without the wrapper there is no swallow to translate, so even a None
    # return value is passed through untouched.
    assert play_with_timeout(_feature_config(enabled=False), lambda: None) is None
