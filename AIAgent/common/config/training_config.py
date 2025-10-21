from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class TrainingConfig:
    dynamic_dataset: bool
    train_percentage: float
    threshold_coverage: int
    load_to_cpu: bool
    epochs: int
    threshold_steps_number: Optional[int] = Field(default=None)
    lr: float = Field(default=0.001)
    batch_size: int = Field(default=16)
    num_hops_1: int = Field(default=5)
    num_hops_2: int = Field(default=10)
    num_of_state_features: int = Field(default=49)
    hidden_channels: int = Field(default=75)
    num_pc_layers: int = Field(default=5)
    normalization: bool = Field(default=True)
    early_stopping_state_len: int = Field(default=5)
    tolerance: float = Field(default=0.0001)
