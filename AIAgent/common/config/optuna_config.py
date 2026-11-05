from enum import Enum
from typing import Optional

from common.config.base import CONFIG
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


class OptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@pydantic_dataclass(config=CONFIG)
class OptunaConfig:
    n_startup_trials: int
    n_trials: int
    n_jobs: int
    study_direction: OptimizationDirection
    trial_uri: Optional[str] = Field(default=None)
