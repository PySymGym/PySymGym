from dataclasses import dataclass
from typing import Callable, TypeAlias

from config import FeatureConfig
from connection.game_server_conn.unsafe_json import asdict
from dataclasses_json import dataclass_json

WSUrl: TypeAlias = str
Undefined: TypeAlias = None


@dataclass_json
@dataclass(slots=True, frozen=True)
class ServerInstanceInfo:
    svm_name: str
    port: int
    ws_url: WSUrl
    pid: int | Undefined


@dataclass_json
@dataclass
class SVMInfo:
    name: str
    count: int
    launch_command: str
    min_port: int
    max_port: int
    server_working_dir: str


def custom_encoder_if_disable_message_checks() -> Callable | None:
    return asdict if FeatureConfig.DISABLE_MESSAGE_CHECKS else None
