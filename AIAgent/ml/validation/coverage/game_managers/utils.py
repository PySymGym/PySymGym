from typing import Callable, TypeVar

from common.classes import GameFailed
from common.config.feature_config import FeatureConfig
from func_timeout import func_set_timeout  # type: ignore

T = TypeVar("T")


def set_timeout_if_needed(
    feature_config: FeatureConfig, func: Callable[..., T]
) -> Callable[..., T]:
    # Applied at call time (not as an import-time decorator) so the timeout
    # comes from the run's feature config, which survives the spawn-based
    # mp.Pool workers as a pickled instance attribute.
    return (
        func_set_timeout(feature_config.save_if_fail_or_timeout.timeout_sec)(func)
        if feature_config.save_if_fail_or_timeout.enabled
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
