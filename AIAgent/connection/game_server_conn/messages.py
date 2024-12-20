import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from common.game import GameMap, GameState, Reward
from config import FeatureConfig
from connection.game_server_conn.unsafe_json import obj_from_dict
from dataclasses_json import config, dataclass_json


class ClientMessageType(str, Enum):
    START = "start"
    STEP = "step"
    STOP = "stop"


@dataclass_json
@dataclass(slots=True)
class ClientMessageBody:
    def type(self) -> ClientMessageType:
        pass


@dataclass_json
@dataclass(slots=True)
class StartMessageBody(ClientMessageBody, GameMap):
    def type(self) -> ClientMessageType:
        return ClientMessageType.START


@dataclass_json
@dataclass(slots=True)
class StepMessageBody(ClientMessageBody):
    StateId: int
    PredictedStateUsefulness: float

    def type(self) -> ClientMessageType:
        return ClientMessageType.STEP


@dataclass_json
@dataclass(slots=True)
class ClientMessage:
    MessageType: str = field(init=False)
    MessageBody: ClientMessageBody = field(
        metadata=config(
            encoder=lambda x: (
                x.to_json() if issubclass(type(x), ClientMessageBody) else json.dumps(x)
            )
        )
    )

    def __post_init__(self):
        self.MessageType = self.MessageBody.type()


class ServerMessageType(str, Enum):
    GAMEOVER = "GameOver"
    MOVE_REVARD = "MoveReward"
    INCORRECT_PREDICTED_STATEID = "IncorrectPredictedStateId"
    READY_FOR_NEXT_STEP = "ReadyForNextStep"
    SERVER_ERROR = "ServerError"


@dataclass_json
@dataclass(slots=True)
class ServerMessage:
    MessageType: ServerMessageType

    class DeserializationException(Exception):
        pass

    def from_json_handle(data, expected):
        if FeatureConfig.DISABLE_MESSAGE_CHECKS:
            return obj_from_dict(json.loads(data))

        try:
            return expected.from_json(data)
        except Exception as e:
            err_to_display = f"{type(e)} - {e}: tried to decode {expected}, got unmatched structure, registered to app.log under [ERROR] tag"
            error = f"{type(e)} - {e}: tried to decode {expected}, got raw data: {json.dumps(json.loads(data), indent=2)}"
            logging.error(error)
            raise ServerMessage.DeserializationException(err_to_display) from e


@dataclass(slots=True)
class GameStateServerMessage(ServerMessage):
    MessageBody: GameState


@dataclass(slots=True)
class RewardServerMessage(ServerMessage):
    MessageBody: Reward


@dataclass(slots=True)
class GameOverServerMessageBody:
    ActualCoverage: Optional[int]
    TestsCount: int
    ErrorsCount: int


@dataclass(slots=True)
class GameOverServerMessage(ServerMessage):
    MessageBody: GameOverServerMessageBody
