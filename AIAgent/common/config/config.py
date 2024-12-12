from typing import Optional

from common.config.mlflow_config import MLFlowConfig
from common.config.optuna_config import OptunaConfig
from common.config.training_config import TrainingConfig
from common.config.validation_config import ValidationConfig
from pydantic import Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class Config:
    optuna_config: OptunaConfig = Field(alias="OptunaConfig")
    training_config: TrainingConfig = Field(alias="TrainingConfig")
    validation_config: ValidationConfig = Field(alias="ValidationConfig")
    mlflow_config: MLFlowConfig = Field(alias="MLFlowConfig")
    weights_uri: Optional[str] = Field(default=None)

    @field_validator("weights_uri", mode="after")
    @classmethod
    def check_if_both_none_or_not_none(cls, weights_uri: str, val_info: ValidationInfo):
        trial_uri = val_info.data["optuna_config"].trial_uri
        if (weights_uri is None and trial_uri is None) or (
            weights_uri is not None and trial_uri is not None
        ):
            return weights_uri
        else:
            raise ValueError(
                "Optuna study's URI and weights URI can be either None or not None at the same time."
            )
