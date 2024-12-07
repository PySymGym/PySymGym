from abc import ABC
from typing import Any

from common.typealias import SVMName
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class SVMInfo(ABC):
    name: SVMName
    launch_command: str
    server_working_dir: str

    def to_dict(
        self,
    ) -> dict[str, Any]:  # GameMap class requires the to_dict method for all its fields
        return self.__dict__


@pydantic_dataclass(config=dict(extra="ignore"))  # type: ignore
class SVMInfoViaServer(SVMInfo):
    min_port: int
    max_port: int
