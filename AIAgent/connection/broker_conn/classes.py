from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Callable, TypeAlias
from pydantic.dataclasses import dataclass as pydantic_dataclass

from common.typealias import SVMName
from config import FeatureConfig
from connection.game_server_conn.unsafe_json import asdict

WSUrl: TypeAlias = str
Undefined: TypeAlias = None


@dataclass_json
@dataclass(slots=True, frozen=True)
class ServerInstanceInfo:
    svm_name: SVMName
    port: int
    ws_url: WSUrl
    pid: int | Undefined


@pydantic_dataclass(config=dict(extra="ignore"))
class SVMInfo:
    name: SVMName
    launch_command: str
    min_port: int
    max_port: int
    server_working_dir: str

    def to_dict(self):  # GameMap class requires the to_dict method for all its fields
        return self.__dict__


def custom_encoder_if_disable_message_checks() -> Callable | None:
    return asdict if FeatureConfig.DISABLE_MESSAGE_CHECKS else None
