from typing import Any

from common.typealias import SVMName
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class SVMInfo:
    name: SVMName
    launch_command: str
    server_working_dir: str
    min_port: int
    max_port: int

    def to_dict(
        self,
    ) -> dict[str, Any]:  # GameMap class requires the to_dict method for all its fields
        return self.__dict__
