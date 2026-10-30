from typing import Any

from common.typealias import SVMName
from common.utils import require_absolute_path
from pydantic import field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class SVMInfo:
    name: SVMName
    launch_command: str
    server_working_dir: str
    min_port: int
    max_port: int

    @field_validator("server_working_dir", mode="before")
    @classmethod
    def _absolute_server_working_dir(cls, value: str) -> str:
        # The broker launches the server with this as its CWD: a relative
        # value would silently resolve against the broker's CWD, so it is
        # rejected (the config loader anchors it before the model sees it).
        return str(require_absolute_path(value))

    def to_dict(
        self,
    ) -> dict[str, Any]:  # GameMap class requires the to_dict method for all its fields
        return self.__dict__
