from dataclasses import dataclass
from typing import TypeAlias

from common.typealias import SVMName
from dataclasses_json import dataclass_json

WSUrl: TypeAlias = str
Undefined: TypeAlias = None


@dataclass_json
@dataclass(slots=True, frozen=True)
class ServerInstanceInfo:
    svm_name: SVMName
    port: int
    ws_url: WSUrl
    pid: int | Undefined
