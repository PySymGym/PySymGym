from typing import Callable, TypeVar

from common.classes import GameFailed
from config import FeatureConfig
from func_timeout import func_set_timeout  # type: ignore

T = TypeVar("T")


def set_timeout_if_needed(func: Callable[..., T]) -> Callable[..., T]:
    return (
        func_set_timeout(FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.timeout_sec)(func)
        if FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.enabled
        else func
    )  # type: ignore


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
