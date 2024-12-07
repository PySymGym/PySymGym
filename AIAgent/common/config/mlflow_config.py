from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class MLFlowConfig:
    experiment_name: str
    tracking_uri: Optional[str] = Field(default=None)
