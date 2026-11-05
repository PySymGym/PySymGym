from typing import Optional

from common.config.base import CONFIG
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(config=CONFIG)
class TrainingConfig:
    dynamic_dataset: bool
    train_percentage: float
    threshold_coverage: int
    load_to_cpu: bool
    epochs: int
    seed: int
    threshold_steps_number: Optional[int] = Field(default=None)
