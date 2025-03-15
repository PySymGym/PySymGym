import socket
import subprocess
from dataclasses import dataclass
from typing import Optional, Union

from common.classes import GameFailed, GameResult
from common.game import GameState
from dataclasses_json import dataclass_json
from torch_geometric.data.hetero_data import HeteroData


@dataclass
class SVMConnectionInfo:
    occupied_port: int
    socket: socket.socket


@dataclass
class GameResultDetails:
    game_result: GameResult
    svm_connection_info: SVMConnectionInfo


@dataclass
class GameFailedDetails:
    game_failed: GameFailed


@dataclass
class ModelGameMapInfo:
    total_game_state: Optional[GameState]
    total_steps: list[HeteroData]
    proc: Optional[subprocess.Popen]
    game_result: Union[GameResultDetails, GameFailedDetails, None]


@dataclass_json
@dataclass(slots=True)
class ModelGameStep:
    GameState: GameState
    Output: list[list[float]]
