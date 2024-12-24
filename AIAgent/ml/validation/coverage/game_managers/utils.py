from typing import Callable, TypeVar

from config import FeatureConfig
from func_timeout import func_set_timeout  # type: ignore

T = TypeVar("T")


def set_timeout_if_needed(func: Callable[..., T]) -> Callable[..., T]:
    return (
        func_set_timeout(FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.timeout_sec)(func)
        if FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.enabled
        else func
    )  # type: ignore
