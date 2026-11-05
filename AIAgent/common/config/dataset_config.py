from pathlib import Path

from common.utils import require_absolute_path
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG: ConfigDict = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class DatasetConfig:
    dataset_base_path: Path  # path to dir with explored dlls
    dataset_description: Path  # full paths to JSON-file with dataset description

    @field_validator("dataset_base_path", "dataset_description", mode="before")
    @classmethod
    def transform(cls, input: str | Path) -> Path:
        return require_absolute_path(input)
