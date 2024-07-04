from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from connection.broker_conn.classes import SVMInfo
from common.typealias import PlatformName


@pydantic_dataclass
class DatasetConfig:
    dataset_base_path: Path  # path to dir with explored dlls
    dataset_description: Path  # full paths to JSON-file with dataset description

    @field_validator("dataset_base_path", "dataset_description", mode="before")
    @classmethod
    def transform(cls, input: str) -> Path:
        return Path(input).resolve()


@pydantic_dataclass
class Platform:
    name: PlatformName
    DatasetConfigs: list[DatasetConfig]
    SVMSInfo: list[SVMInfo]


@pydantic_dataclass
class ServersConfig:
    Platforms: list[Platform]
    count: int


@pydantic_dataclass
class OptunaConfig:
    n_startup_trials: (
        int  # number of initial trials with random sampling for optuna's TPESampler
    )
    n_trials: int  # number of optuna's trials
    n_jobs: int
    study_direction: str


@pydantic_dataclass
class TrainingConfig:
    dynamic_dataset: bool
    train_percentage: float
    threshold_coverage: int
    load_to_cpu: bool
    epochs: int
    threshold_steps_number: Optional[int] = Field(default=None)


@pydantic_dataclass
class Config:
    ServersConfig: ServersConfig
    OptunaConfig: OptunaConfig
    TrainingConfig: TrainingConfig
    path_to_weights: Optional[Path] = Field(default=None)

    @field_validator("path_to_weights", mode="before")
    @classmethod
    def transform(cls, input: Optional[str]) -> Optional[Path]:
        return Path(input).absolute() if input is not None else None
