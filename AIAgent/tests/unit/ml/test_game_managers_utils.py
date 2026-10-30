"""Unit tests for the shared game-manager utilities.

``play_with_timeout`` is tested against the real worker-thread wrapper: the
inner-timeout case relies on the worker re-raising a ``FunctionTimedOut``
raised inside the wrapped function (it propagates directly, unlike
``func_timeout``'s swallow-and-return-``None`` behaviour, #672).
"""

import time

import pytest
from common.config.feature_config import FeatureConfig, SaveIfFailOrTimeout
from func_timeout import FunctionTimedOut
from ml.validation.coverage.game_managers import utils
from ml.validation.coverage.game_managers.utils import play_with_timeout

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def _feature_config(**overrides) -> FeatureConfig:
    return FeatureConfig(save_if_fail_or_timeout=SaveIfFailOrTimeout(**overrides))


def test_play_with_timeout_passes_the_result_through() -> None:
    assert play_with_timeout(_feature_config(), lambda: (1, 2)) == (1, 2)


def test_play_with_timeout_propagates_an_inner_timeout() -> None:
    # A FunctionTimedOut raised inside the wrapped function propagates
    # directly (the worker re-raises it), not translated from a None return
    # (#672's swallow quirk is gone).
    def inner():
        raise FunctionTimedOut(timedOutAfter=1800)

    with pytest.raises(FunctionTimedOut) as excinfo:
        play_with_timeout(_feature_config(), inner)

    assert excinfo.value.timedOutAfter == 1800


def test_play_with_timeout_raises_on_a_none_return() -> None:
    # The game-play functions contractually return a non-None result, so a
    # None return under an active timeout is always a timeout signal.
    with pytest.raises(FunctionTimedOut):
        play_with_timeout(_feature_config(), lambda: None)


def test_play_with_timeout_passes_through_when_disabled() -> None:
    # Without the timeout there is no worker thread, so even a None return
    # value is passed through untouched.
    assert play_with_timeout(_feature_config(enabled=False), lambda: None) is None


def test_play_with_timeout_propagates_a_plain_exception() -> None:
    # A non-timeout exception from the worker propagates unwrapped.
    def inner():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        play_with_timeout(_feature_config(), inner)


def test_wall_clock_timeout_tracks_the_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tracker starts empty so the assertions are exact.
    monkeypatch.setattr(utils, "_timed_out_threads", set(), raising=False)

    def slow():
        time.sleep(1.3)

    with pytest.raises(FunctionTimedOut) as excinfo:
        play_with_timeout(_feature_config(timeout_sec=1), slow)

    assert excinfo.value.timedOutAfter == 1
    # The timed-out worker is tracked, not silently orphaned.
    assert len(utils._timed_out_threads) == 1
    worker = next(iter(utils._timed_out_threads))
    assert worker.is_alive()
    # Once the worker finishes, pruning empties the tracker.
    worker.join(timeout=10)
    assert utils._prune_finished_threads() == 0
    assert utils._timed_out_threads == set()
