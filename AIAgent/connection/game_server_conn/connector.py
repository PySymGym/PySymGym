import logging
from functools import wraps
from typing import Optional

import websocket
from common.config.feature_config import FeatureConfig
from common.game import GameMap, GameState
from connection.errors_connection import ConnectionLostError

from .messages import (
    ClientMessage,
    GameOverServerMessage,
    GameStateServerMessage,
    IncorrectPredictedStateIdBody,
    Reward,
    RewardServerMessage,
    ServerMessage,
    ServerMessageType,
    StartMessageBody,
    StepMessageBody,
)


class Connector:
    class GameOver(Exception):
        def __init__(
            self,
            actual_coverage: Optional[int],
            tests_count: int,
            steps_count: int,
            errors_count: int,
            *args,
        ) -> None:
            self.actual_coverage = actual_coverage
            self.tests_count = tests_count
            self.steps_count = steps_count
            self.errors_count = errors_count
            super().__init__(*args)

    def __init__(
        self,
        ws: websocket.WebSocket,
        map: GameMap,
        feature_config: FeatureConfig,
    ) -> None:
        self.ws = ws
        self._disable_message_checks = feature_config.disable_message_checks

        start_message = ClientMessage(StartMessageBody.from_game_map(map))
        logging.debug(f"--> StartMessage  : {start_message}")
        self.send(start_message.to_json())
        self._current_step = 0
        self.game_is_over = False
        self._gameover: Connector.GameOver | None = None
        self.map = map

    def catch_losing_of_connection(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (websocket.WebSocketException, OSError) as e:
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
            raise self._gameover

        matching_message_type = ServerMessage.from_json_handle(
            msg,
            expected=ServerMessage,
            disable_message_checks=self._disable_message_checks,
        ).MessageType
        match matching_message_type:
            case ServerMessageType.GAMEOVER:
                deser_msg = GameOverServerMessage.from_json_handle(
                    msg,
                    expected=GameOverServerMessage,
                    disable_message_checks=self._disable_message_checks,
                )
                self.game_is_over = True
                self._gameover = Connector.GameOver(
                    actual_coverage=deser_msg.MessageBody.ActualCoverage,
                    tests_count=deser_msg.MessageBody.TestsCount,
                    steps_count=deser_msg.MessageBody.StepsCount,
                    errors_count=deser_msg.MessageBody.ErrorsCount,
                )
                logging.debug(f"--> {matching_message_type}")
                raise self._gameover
            case _:
                return msg

    def recv_state_or_throw_gameover(self) -> GameState:
        received = self.receive()
        data = GameStateServerMessage.from_json_handle(
            self._raise_if_gameover(received),
            expected=GameStateServerMessage,
            disable_message_checks=self._disable_message_checks,
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

    def recv_reward_or_throw_gameover(self) -> Reward | IncorrectPredictedStateIdBody:
        received = self.receive()
        decoded = RewardServerMessage.from_json_handle(
            self._raise_if_gameover(received),
            expected=RewardServerMessage,
            disable_message_checks=self._disable_message_checks,
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

            case ServerMessageType.MOVE_REWARD:
                self._current_step += 1
                return msg.MessageBody

            case _:
                raise RuntimeError(
                    f"Unexpected message type received: {msg.MessageType}"
                )
