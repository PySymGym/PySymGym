from typing import Any, Callable

from config import FeatureConfig
from func_timeout import func_set_timeout  # type: ignore


def set_timeout_if_needed(func: Callable[..., Any]) -> Callable[..., Any]:
    return (
        func_set_timeout(FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.timeout_sec)(func)
        if FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.enabled
        else func
    )  # type: ignore
