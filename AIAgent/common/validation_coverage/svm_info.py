from typing import Any, Optional

from common.typealias import SVMName
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class SVMInfo:
    name: SVMName
    launch_command: str
    server_working_dir: str
    min_port: Optional[int] = Field(default=None)
    max_port: Optional[int] = Field(default=None)

    def to_dict(
        self,
    ) -> dict[str, Any]:  # GameMap class requires the to_dict method for all its fields
        return self.__dict__
