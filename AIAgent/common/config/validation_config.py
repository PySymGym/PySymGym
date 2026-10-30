from abc import ABC
from typing import Literal, Union

from common.config.platform_config import Platform
from common.dataset import TrainingDatasetMode
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class ValidationMode:
    pass


@pydantic_dataclass(config=_CONFIG)
class CriterionValidation(ValidationMode):
    val_type: Literal["loss"]
    batch_size: int
    dataset: TrainingDatasetMode = Field(default=TrainingDatasetMode.VALIDATION)


@pydantic_dataclass(config=_CONFIG)
class SVMValidation(ValidationMode, ABC):
    platforms_config: list[Platform] = Field(alias="PlatformsConfig")
    process_count: int = Field()
    fail_immediately: bool = Field(default=False)


@pydantic_dataclass(config=_CONFIG)
class SVMValidationSendEachStep(SVMValidation):
    val_type: Literal["svms_each_step"] = Field()  # type: ignore


@pydantic_dataclass(config=_CONFIG)
class SVMValidationSendModel(SVMValidation):
    val_type: Literal["svms_model"] = Field()  # type: ignore


@pydantic_dataclass(config=_CONFIG)
class CustomValidation(ValidationMode):
    val_type: Literal["custom"] = Field()
    val_sequence: list[
        Union[CriterionValidation, SVMValidationSendEachStep, SVMValidationSendModel]
    ] = Field()
    process_count: int = Field()


@pydantic_dataclass(config=_CONFIG)
class ValidationConfig:
    validation_mode: Union[
        CriterionValidation,
        SVMValidationSendEachStep,
        SVMValidationSendModel,
        CustomValidation,
    ] = Field(discriminator="val_type")
