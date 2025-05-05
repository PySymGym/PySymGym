from abc import ABC
from typing import Literal, Optional, Union

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
    dataset: Optional[Literal["validation", "training"]] = Field(default="validation")


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
class CustomValidation(ValidationMode):
    val_type: Literal["custom"] = Field()
    val_sequence: list[
        Union[CriterionValidation, SVMValidationSendEachStep, SVMValidationSendModel]
    ] = Field()
    process_count: int = Field()


@pydantic_dataclass
class ValidationConfig:
    validation_mode: Union[
        CriterionValidation,
        SVMValidationSendEachStep,
        SVMValidationSendModel,
        CustomValidation,
    ] = Field(discriminator="val_type")
