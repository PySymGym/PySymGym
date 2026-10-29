import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Optional, TypeVar

from common.game import GameMap, GameState, Reward
from connection.errors_connection import ProtocolVersionMismatchError
from connection.game_server_conn.unsafe_json import BareObject, obj_from_dict
from dataclasses_json import config, dataclass_json

_T = TypeVar("_T")

#: The protocol version the agent speaks. Version 1 is the historical
#: format without the ``Version`` field (what the V# and usvm servers
#: speak today); version 2 adds the ``Version`` field to the envelope.
#: The wire contract lives in ``docs/integration.rst``.
PROTOCOL_VERSION: Final[int] = 2

_legacy_version_notice_issued = False


def check_protocol_version(envelope: object) -> int | None:
    """Validate the ``Version`` field of a received message envelope.

    Returns the observed version, or ``None`` for a legacy v1 envelope
    (no ``Version`` field) or a non-dict envelope (structural validation
    handles those). Raises :class:`ProtocolVersionMismatchError` when the
    field is present with a foreign version. A legacy envelope logs a loud
    warning once per process.
    """
    global _legacy_version_notice_issued
    if not isinstance(envelope, dict):
        return None
    version = envelope.get("Version")
    if version is None:
        if not _legacy_version_notice_issued:
            _legacy_version_notice_issued = True
            logging.warning(
                "game-server message carries no Version field: the server "
                "speaks the legacy protocol (v1); this client speaks "
                "v%s (wire contract: docs/integration.rst)",
                PROTOCOL_VERSION,
            )
        return None
    if version != PROTOCOL_VERSION:
        raise ProtocolVersionMismatchError(
            f"the game server speaks protocol version {version}; "
            f"this client speaks v{PROTOCOL_VERSION} "
            "(wire contract: docs/integration.rst)"
        )
    return version


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
class StartMessageBody(ClientMessageBody):
    # The wire body of the ``start`` message. Mirrors
    # :class:`common.game.GameMap` field-for-field (same names, same
    # order); the sync test in ``test_messages.py`` keeps the two in
    # lockstep so a domain-model change cannot silently change the wire
    # format. The multiple inheritance that fused the wire body with the
    # domain model was replaced by this explicit mapping (#582).
    StepsToPlay: int
    StepsToStart: int
    AssemblyFullName: str
    NameOfObjectToCover: str
    DefaultSearcher: str
    MapName: str

    def type(self) -> ClientMessageType:
        return ClientMessageType.START

    @classmethod
    def from_game_map(cls, game_map: GameMap) -> "StartMessageBody":
        return cls(
            StepsToPlay=game_map.StepsToPlay,
            StepsToStart=game_map.StepsToStart,
            AssemblyFullName=game_map.AssemblyFullName,
            NameOfObjectToCover=game_map.NameOfObjectToCover,
            DefaultSearcher=game_map.DefaultSearcher,
            MapName=game_map.MapName,
        )


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

    @staticmethod
    def from_json_handle(
        data: str, expected: type[_T], disable_message_checks: bool
    ) -> _T | BareObject:
        try:
            envelope = json.loads(data)
        except json.JSONDecodeError as e:
            raise ServerMessage.DeserializationException(f"invalid JSON: {e}") from e

        # The version check is a protocol-level invariant: it runs
        # regardless of disable_message_checks.
        check_protocol_version(envelope)

        if disable_message_checks:
            return obj_from_dict(envelope)

        try:
            return expected.from_dict(envelope)
        except Exception as e:
            err_to_display = f"{type(e)} - {e}: tried to decode {expected}, got unmatched structure, registered to app.log under [ERROR] tag"
            error = f"{type(e)} - {e}: tried to decode {expected}, got raw data: {json.dumps(envelope, indent=2)}"
            logging.error(error)
            raise ServerMessage.DeserializationException(err_to_display) from e


@dataclass(slots=True)
class GameStateServerMessage(ServerMessage):
    MessageBody: GameState


@dataclass_json
@dataclass(slots=True)
class IncorrectPredictedStateIdBody:
    StateId: int


@dataclass(slots=True)
class RewardServerMessage(ServerMessage):
    # The reward channel carries two wire message types: MoveReward (a
    # Reward body) and IncorrectPredictedStateId (a StateId body). The
    # required fields are disjoint, so the union decodes unambiguously.
    MessageBody: Reward | IncorrectPredictedStateIdBody


@dataclass(slots=True)
class GameOverServerMessageBody:
    ActualCoverage: Optional[int]
    TestsCount: int
    StepsCount: int
    ErrorsCount: int


@dataclass(slots=True)
class GameOverServerMessage(ServerMessage):
    MessageBody: GameOverServerMessageBody
