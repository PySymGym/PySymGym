import logging
from typing import Optional

from common.config.broker_config import BrokerConfig
from common.config.connector_config import GameServerConnectorConfig
from common.config.feature_config import FeatureConfig
from common.config.mlflow_config import MLFlowConfig
from common.config.optuna_config import OptunaConfig
from common.config.training_config import TrainingConfig
from common.config.validation_config import ValidationConfig
from pydantic import Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class Config:
    optuna_config: OptunaConfig = Field(alias="OptunaConfig")
    training_config: TrainingConfig = Field(alias="TrainingConfig")
    validation_config: ValidationConfig = Field(alias="ValidationConfig")
    mlflow_config: MLFlowConfig = Field(alias="MLFlowConfig")
    weights_uri: Optional[str] = Field(default=None)
    # Runtime/infrastructure settings merged from the old AIAgent/config.py
    # constant classes; each defaults to the previous hardcoded value so a
    # config that omits them keeps the historical behavior.
    broker_config: BrokerConfig = Field(
        alias="BrokerConfig", default_factory=BrokerConfig
    )
    game_server_connector_config: GameServerConnectorConfig = Field(
        alias="GameServerConnectorConfig",
        default_factory=GameServerConnectorConfig,
    )
    feature_config: FeatureConfig = Field(
        alias="FeatureConfig", default_factory=FeatureConfig
    )
    logger_level: int = Field(default=logging.INFO)

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
