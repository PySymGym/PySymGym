from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class MLFlowConfig:
    experiment_name: str
    tracking_uri: Optional[str] = Field(default=None)
