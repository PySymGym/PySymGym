from dataclasses import dataclass
from typing import TypeAlias

from dataclasses_json import dataclass_json

WSUrl: TypeAlias = str
Undefined: TypeAlias = None


@dataclass_json
@dataclass(slots=True, frozen=True)
class ServerInstanceInfo:
    port: int
    ws_url: WSUrl
    pid: int | Undefined
