from typing import Optional
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from enum import Enum


class OptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@pydantic_dataclass
class OptunaConfig:
    n_startup_trials: int
    n_trials: int
    n_jobs: int
    study_direction: OptimizationDirection
    trial_uri: Optional[str] = Field(default=None)
