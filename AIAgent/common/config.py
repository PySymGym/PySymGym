from pathlib import Path
from typing import Literal, Optional, Union

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
    dataset_configs: list[DatasetConfig] = Field(alias="DatasetConfigs")
    svms_info: list[SVMInfo] = Field(alias="SVMSInfo")


@pydantic_dataclass
class OptunaConfig:
    n_startup_trials: int
    n_trials: int
    n_jobs: int
    study_direction: str
    path_to_study: Optional[Path] = Field(default=None)


@pydantic_dataclass
class TrainingConfig:
    dynamic_dataset: bool
    train_percentage: float
    threshold_coverage: int
    load_to_cpu: bool
    epochs: int
    threshold_steps_number: Optional[int] = Field(default=None)


@pydantic_dataclass
class ValidationWithLoss:
    val_type: Literal["loss"]
    batch_size: int


@pydantic_dataclass
class ValidationWithSVMs:
    val_type: Literal["svms"]
    servers_count: int


@pydantic_dataclass
class ValidationConfig:
    validation: Union[ValidationWithLoss, ValidationWithSVMs] = Field(discriminator="val_type")


@pydantic_dataclass
class MLFlowConfig:
    experiment_name: str
    tracking_uri: Optional[str] = Field(default=None)


@pydantic_dataclass
class Config:
    platforms_config: list[Platform] = Field(alias="PlatformsConfig")
    optuna_config: OptunaConfig = Field(alias="OptunaConfig")
    training_config: TrainingConfig = Field(alias="TrainingConfig")
    validation_config: ValidationConfig = Field(alias="ValidationConfig")
    mlflow_config: MLFlowConfig = Field(alias="MLFlowConfig")
    path_to_weights: Optional[Path] = Field(default=None)

    @field_validator("path_to_weights", mode="before")
    @classmethod
    def transform(cls, input: Optional[str]) -> Optional[Path]:
        return Path(input).resolve() if input is not None else None

    @field_validator("optuna_config", mode="after")
    @classmethod
    def check_if_both_none(cls, optuna_config: OptunaConfig):
        if (
            cls.path_to_weights is None
            and optuna_config.path_to_study is None
            or cls.path_to_weights is not None
            and optuna_config.path_to_study is not None
        ):
            return optuna_config
        else:
            raise ValueError(
                "Path to optuna study and path to weights can be either None or not None at the same time."
            )
