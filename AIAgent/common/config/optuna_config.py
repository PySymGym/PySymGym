from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class OptunaConfig:
    n_startup_trials: int
    n_trials: int
    n_jobs: int
    study_direction: str
    trial_uri: Optional[str] = Field(default=None)
