from pathlib import Path

from common.utils import require_absolute_path
from pydantic import field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class DatasetConfig:
    dataset_base_path: Path  # path to dir with explored dlls
    dataset_description: Path  # full paths to JSON-file with dataset description

    @field_validator("dataset_base_path", "dataset_description", mode="before")
    @classmethod
    def transform(cls, input: str | Path) -> Path:
        return require_absolute_path(input)
