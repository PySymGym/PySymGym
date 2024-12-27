from abc import ABC
from typing import Literal, Union

from common.config.platform_config import Platform
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class ValidationMode:
    pass


@pydantic_dataclass
class CriterionValidation(ValidationMode):
    val_type: Literal["loss"]
    batch_size: int


@pydantic_dataclass
class SVMValidation(ValidationMode, ABC):
    platforms_config: list[Platform] = Field(alias="PlatformsConfig")
    process_count: int = Field()
    fail_immediately: bool = Field(default=False)


@pydantic_dataclass
class SVMValidationSendEachStep(SVMValidation):
    val_type: Literal["svms_each_step"] = Field()  # type: ignore


@pydantic_dataclass
class SVMValidationSendModel(SVMValidation):
    val_type: Literal["svms_model"] = Field()  # type: ignore


@pydantic_dataclass
class ValidationConfig:
    validation_mode: Union[
        CriterionValidation, SVMValidationSendEachStep, SVMValidationSendModel
    ] = Field(discriminator="val_type")
