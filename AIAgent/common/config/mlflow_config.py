from typing import Optional

from common.config.base import CONFIG
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(config=CONFIG)
class MLFlowConfig:
    experiment_name: str
    tracking_uri: Optional[str] = Field(default=None)
