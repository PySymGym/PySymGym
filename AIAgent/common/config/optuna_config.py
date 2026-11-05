from typing import Optional
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from enum import Enum

_CONFIG = {"extra": "forbid", "populate_by_name": True}


class OptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@pydantic_dataclass(config=_CONFIG)
class OptunaConfig:
    n_startup_trials: int
    n_trials: int
    n_jobs: int
    study_direction: OptimizationDirection
    trial_uri: Optional[str] = Field(default=None)
