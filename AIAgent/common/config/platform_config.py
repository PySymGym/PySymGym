from common.config.dataset_config import DatasetConfig
from common.typealias import PlatformName
from common.validation_coverage.svm_info import SVMInfo
from pydantic import (
    Field,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class Platform:
    name: PlatformName
    dataset_configs: list[DatasetConfig] = Field(alias="DatasetConfigs")
    svms_info: list[SVMInfo] = Field(alias="SVMSInfo")
