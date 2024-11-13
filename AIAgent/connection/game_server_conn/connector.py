from functools import wraps
import logging
import logging.config
from typing import Optional

import websocket
from connection.errors_connection import ConnectionLostError
from common.game import GameMap, GameState

from .messages import (
    ClientMessage,
    GameOverServerMessage,
    GameStateServerMessage,
    Reward,
    RewardServerMessage,
    ServerMessage,
    ServerMessageType,
    StartMessageBody,
    StepMessageBody,
)


class Connector:
    class WrongConnectorStateError(Exception):
        def __init__(
            self, source: str, received: str, expected: str, at_step: int
        ) -> None:
            super().__init__(
                f"Wrong operations order at step #{at_step}: at function \
                <{source}> received {received}, expected {expected}",
            )

    class IncorrectSentStateError(Exception):
        pass

    class GameOver(Exception):
        def __init__(
            self,
            actual_coverage: Optional[int],
            tests_count: int,
            errors_count: int,
            *args,
        ) -> None:
            self.actual_coverage = actual_coverage
            self.tests_count = tests_count
            self.errors_count = errors_count
            super().__init__(*args)

    def __init__(
        self,
        ws: websocket.WebSocket,
        map: GameMap,
    ) -> None:
        self.ws = ws

        start_message = ClientMessage(StartMessageBody(**map.to_dict()))
        logging.debug(f"--> StartMessage  : {start_message}")
        self.send(start_message.to_json())
        self._current_step = 0
        self.game_is_over = False
        self.map = map

    def catch_losing_of_connection(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ConnectionResetError as e:
                raise ConnectionLostError from e

        return wrapper

    @catch_losing_of_connection
    def receive(self):
        return self.ws.recv()

    @catch_losing_of_connection
    def send(self, msg):
        return self.ws.send(msg)

    def _raise_if_gameover(self, msg) -> GameOverServerMessage | str:
        if self.game_is_over:
            raise Connector.GameOver

        matching_message_type = ServerMessage.from_json_handle(
            msg, expected=ServerMessage
        ).MessageType
        match matching_message_type:
            case ServerMessageType.GAMEOVER:
                deser_msg = GameOverServerMessage.from_json_handle(
                    msg, expected=GameOverServerMessage
                )
                self.game_is_over = True
                logging.debug(f"--> {matching_message_type}")
                raise Connector.GameOver(
                    actual_coverage=deser_msg.MessageBody.ActualCoverage,
                    tests_count=deser_msg.MessageBody.TestsCount,
                    errors_count=deser_msg.MessageBody.ErrorsCount,
                )
            case _:
                return msg

    def recv_state_or_throw_gameover(self) -> GameState:
        received = self.receive()
        data = GameStateServerMessage.from_json_handle(
            self._raise_if_gameover(received),
            expected=GameStateServerMessage,
        )
        logging.debug(f"<-- {data.MessageType}")
        return data.MessageBody

    def send_step(self, next_state_id: int, predicted_usefullness: int):
        do_step_message = ClientMessage(
            StepMessageBody(
                StateId=next_state_id, PredictedStateUsefulness=predicted_usefullness
            )
        )
        logging.debug(f"--> ClientMessage : {do_step_message}")
        self.send(do_step_message.to_json())
        self._sent_state_id = next_state_id

    def recv_reward_or_throw_gameover(self) -> Reward:
        received = self.receive()
        decoded = RewardServerMessage.from_json_handle(
            self._raise_if_gameover(received),
            expected=RewardServerMessage,
        )
        logging.debug(f"<-- MoveReward    : {decoded.MessageBody}")

        return self._process_reward_server_message(decoded)

    def _process_reward_server_message(self, msg):
        match msg.MessageType:
            case ServerMessageType.INCORRECT_PREDICTED_STATEID:
                logging.error(
                    f"Sending state_id={self._sent_state_id} for map {self.map.MapName} at step #{self._current_step} resulted in {msg.MessageType}"
                )
                return msg.MessageBody

            case ServerMessageType.MOVE_REVARD:
                self._current_step += 1
                return msg.MessageBody

            case _:
                raise RuntimeError(
                    f"Unexpected message type received: {msg.MessageType}"
                )
