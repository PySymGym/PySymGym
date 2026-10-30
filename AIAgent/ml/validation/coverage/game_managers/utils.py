import logging
import threading
from typing import Any, Callable, TypeVar

from common.classes import GameFailed
from common.config.feature_config import FeatureConfig
from func_timeout import FunctionTimedOut  # type: ignore

T = TypeVar("T")

# Worker threads that hit the wall-clock timeout and are still alive. An
# in-thread raise (func_timeout's stop mechanism) cannot interrupt a blocked
# C-level socket recv, so a timed-out worker keeps running until the
# connector's recv timeout (response_timeout_sec) fires. Tracking the worker
# (instead of silently orphaning it) makes the leak observable and bounded:
# each entry self-terminates within that recv timeout and is pruned on the
# next timeout.
_timed_out_threads: set[threading.Thread] = set()
_timed_out_threads_lock = threading.Lock()


def _prune_finished_threads() -> int:
    """Drop finished workers from the tracker; return the count still alive."""
    with _timed_out_threads_lock:
        for thread in [t for t in _timed_out_threads if not t.is_alive()]:
            _timed_out_threads.discard(thread)
        return len(_timed_out_threads)


def play_with_timeout(
    feature_config: FeatureConfig, func: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Call ``func`` under the run's per-map timeout.

    With the timeout feature enabled, ``func`` runs in a daemon worker thread
    and the caller joins it for the configured timeout:

    - Worker finished: its result is returned, or the exception it raised is
      re-raised here — including a ``FunctionTimedOut`` raised from inside
      ``func``, which propagates directly to the caller's timeout branch.
    - Worker still alive (wall-clock timeout): the worker is registered in
      ``_timed_out_threads`` and a ``FunctionTimedOut`` is raised into the
      caller's timeout branch.

    The game-play functions contractually return a non-``None`` result, so a
    ``None`` return under an active timeout is still treated as a timeout
    signal. With the timeout feature disabled, ``func`` is called inline and
    the return value is passed through untouched.
    """
    save_if_fail_or_timeout = feature_config.save_if_fail_or_timeout
    if not save_if_fail_or_timeout.enabled:
        return func(*args, **kwargs)

    timeout_sec = save_if_fail_or_timeout.timeout_sec
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["result"] = func(*args, **kwargs)
        except BaseException as e:
            # BaseException, not Exception: FunctionTimedOut derives from
            # BaseException and must be captured, not crash the worker.
            outcome["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        # Wall-clock timeout: the worker is still running (typically blocked
        # in a socket recv). Track it — it self-terminates within the
        # connector's recv timeout — and report the timeout.
        alive = _prune_finished_threads() + 1
        with _timed_out_threads_lock:
            _timed_out_threads.add(thread)
        logging.warning(
            f"game-play function timed out after {timeout_sec}s; "
            f"tracking its worker thread ({alive} timed-out worker(s) alive)"
        )
        raise FunctionTimedOut(
            msg=f"the game-play function timed out after {timeout_sec} seconds",
            timedOutAfter=timeout_sec,
        )

    if "error" in outcome:
        raise outcome["error"]
    result = outcome.get("result")
    if result is None:
        raise FunctionTimedOut(
            msg="the game-play function timed out internally",
            timedOutAfter=timeout_sec,
        )
    return result


def unexhausted_steps_failure(
    map_name: str, steps_taken: int, steps_expected: int, coverage: float
) -> GameFailed:
    """Build the failure for a game that ended before all steps were exhausted.

    The symbolic engine stopped exploring without reaching 100% coverage and
    without playing all planned steps, which indicates an engine defect rather
    than a model quality issue. The reason names the map and the observed
    numbers so a failed run points straight at the offending map.
    """
    return GameFailed(
        reason=(
            f"Not all steps exhausted on {map_name}: "
            f"{steps_taken} of {steps_expected} steps taken, "
            f"actual coverage {coverage:.2f}% != 100%"
        )
    )
