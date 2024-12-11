from common.config.dataset_config import DatasetConfig
from common.typealias import PlatformName
from common.validation_coverage.svm_info import SVMInfo
from pydantic import (
    Field,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class Platform:
    name: PlatformName
    dataset_configs: list[DatasetConfig] = Field(alias="DatasetConfigs")
    svms_info: list[SVMInfo] = Field(alias="SVMSInfo")
