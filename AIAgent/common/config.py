from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, ValidationInfo, field_validator
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
    study_uri: Optional[str] = Field(default=None)


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
    platforms_config: list[Platform] = Field(alias="PlatformsConfig")
    fail_immediately: bool = Field(default=False)


@pydantic_dataclass
class ValidationConfig:
    validation: Union[ValidationWithLoss, ValidationWithSVMs] = Field(
        discriminator="val_type"
    )


@pydantic_dataclass
class MLFlowConfig:
    experiment_name: str
    tracking_uri: Optional[str] = Field(default=None)


@pydantic_dataclass
class Config:
    optuna_config: OptunaConfig = Field(alias="OptunaConfig")
    training_config: TrainingConfig = Field(alias="TrainingConfig")
    validation_config: ValidationConfig = Field(alias="ValidationConfig")
    mlflow_config: MLFlowConfig = Field(alias="MLFlowConfig")
    weights_uri: Optional[str] = Field(default=None)

    @field_validator("weights_uri", mode="after")
    @classmethod
    def check_if_both_none(cls, weights_uri: str, val_info: ValidationInfo):
        study_uri = val_info.data["optuna_config"].study_uri
        if (weights_uri is None and study_uri is None) or (
            weights_uri is not None and study_uri is not None
        ):
            return weights_uri
        else:
            raise ValueError(
                "Optuna study's URI and weights URI can be either None or not None at the same time."
            )
