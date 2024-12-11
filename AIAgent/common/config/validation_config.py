from abc import ABC
from typing import Literal, Union

from common.config.platform_config import Platform
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class Validation(ABC):
    pass


@pydantic_dataclass
class ValidationLoss(Validation):
    val_type: Literal["loss"]
    batch_size: int


@pydantic_dataclass
class ValidationSVM(Validation, ABC):
    platforms_config: list[Platform] = Field(alias="PlatformsConfig")
    process_count: int = Field()
    fail_immediately: bool = Field(default=False)


@pydantic_dataclass
class ValidationSVMViaServer(ValidationSVM):
    val_type: Literal["svms_server"] = Field()  # type: ignore


@pydantic_dataclass
class ValidationConfig:
    validation: Union[ValidationLoss, ValidationSVMViaServer] = Field(
        discriminator="val_type"
    )
