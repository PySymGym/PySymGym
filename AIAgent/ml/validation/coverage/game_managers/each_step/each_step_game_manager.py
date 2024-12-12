import logging
import traceback
from time import perf_counter
from typing import TypeAlias

import torch
from common.classes import GameFailed, GameResult, Map2Result
from common.game import GameMap2SVM
from config import FeatureConfig
from connection.broker_conn.socket_manager import game_server_socket_manager
from connection.errors_connection import GameInterruptedError
from connection.game_server_conn.connector import Connector
from func_timeout import FunctionTimedOut
from ml.dataset import convert_input_to_tensor
from ml.protocols import Predictor
from ml.validation.coverage.game_managers.base_game_manager import BaseGameManager
from ml.validation.coverage.game_managers.each_step.game_states_utils import (
    get_states,
    update_game_state,
)
from ml.validation.coverage.game_managers.utils import set_timeout_if_needed

TimeDuration: TypeAlias = float


class EachStepGameManager(BaseGameManager):
    def __init__(self, with_predictor: Predictor):
        self.with_predictor = with_predictor
        super().__init__(self)

    @set_timeout_if_needed
    def _play_game_map_with_svm(
        self, game_map2svm: GameMap2SVM
    ) -> tuple[GameResult, TimeDuration]:
        with game_server_socket_manager(game_map2svm.SVMInfo) as ws:
            with_connector = Connector(ws, game_map2svm.GameMap)
            steps_count = 0
            game_state = None
            actual_coverage = None
            steps = with_connector.map.StepsToPlay

            start_time = perf_counter()

            map_steps = []

            def add_single_step(input, output):
                hetero_input, _ = convert_input_to_tensor(input)
                hetero_input["y_true"] = output
                map_steps.append(hetero_input)  # noqa: F821

            try:
                for _ in range(with_connector.map.StepsToPlay):
                    if steps_count == 0:
                        game_state = with_connector.recv_state_or_throw_gameover()
                    else:
                        delta = with_connector.recv_state_or_throw_gameover()
                        game_state = update_game_state(game_state, delta)

                    predicted_state_id, nn_output = self.with_predictor.predict(
                        game_state
                    )

                    add_single_step(game_state, nn_output)

                    logging.debug(
                        f"<{self.with_predictor.name()}> step: {steps_count}, available states: {get_states(game_state)}, predicted: {predicted_state_id}"
                    )

                    with_connector.send_step(
                        next_state_id=predicted_state_id,
                        predicted_usefullness=42.0,  # left it a constant for now
                    )

                    _ = with_connector.recv_reward_or_throw_gameover()
                    steps_count += 1

                _ = with_connector.recv_state_or_throw_gameover()  # wait for gameover
                steps_count += 1
            except Connector.GameOver as gameover:
                if game_state is None:
                    logging.warning(
                        f"<{self.with_predictor.name()}>: immediate GameOver on {with_connector.map.MapName}"
                    )
                    return (
                        GameResult(steps, 0, 0, 0),
                        perf_counter() - start_time,
                    )
                if gameover.actual_coverage is not None:
                    actual_coverage = gameover.actual_coverage

                tests_count = gameover.tests_count
                errors_count = gameover.errors_count

            end_time = perf_counter()
            if actual_coverage != 100 and steps_count != steps:
                logging.warning(
                    f"<{self.with_predictor.name()}>: not all steps exshausted on {with_connector.map.MapName} with non-100% coverage"
                    f"steps taken: {steps_count}, actual coverage: {actual_coverage:.2f}"
                )
                steps_count = steps

            model_result = GameResult(
                steps_count=steps_count,
                tests_count=tests_count,
                errors_count=errors_count,
                actual_coverage_percent=actual_coverage,
            )
            self._game_states[str(game_map2svm.GameMap)] = map_steps
            torch.cuda.empty_cache()
            return model_result, end_time - start_time

    def play_game_map(
        self,
        game_map2svm: GameMap2SVM,
    ) -> Map2Result:
        logging.info(
            f"<{self.with_predictor.name()}> is playing {game_map2svm.GameMap.MapName}"
        )
        need_to_save = False
        try:
            game_result, time = self._play_game_map_with_svm(game_map2svm)
            logging.info(
                f"<{self.with_predictor.name()}> finished map {game_map2svm.GameMap.MapName} "
                f"in {game_result.steps_count} steps, {time} seconds, "
                f"actual coverage: {game_result.actual_coverage_percent:.2f}"
            )
        except FunctionTimedOut as error:
            need_to_save = True
            logging.warning(
                f"<{self.with_predictor.name()}> timeouted on map {game_map2svm.GameMap.MapName} with {error.timedOutAfter}s"
            )
            game_result = GameFailed(reason=type(error))
        except GameInterruptedError as error:
            logging.warning(
                f"<{self.with_predictor.name()}> failed on map {game_map2svm.GameMap.MapName} with {error.__class__.__name__}: {error.desc}"
            )
            game_result = GameFailed(reason=type(error))
        except Exception as error:
            need_to_save = True
            logging.warning(
                (
                    f"<{self.with_predictor.name()}> failed on map {game_map2svm.GameMap.MapName}:\n"
                    + "\n".join(
                        traceback.format_exception(
                            type(error), value=error, tb=error.__traceback__
                        )
                    )
                )
            )
            game_result = GameFailed(reason=type(error))

        if need_to_save:
            FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.save_model(
                self.with_predictor.model(), with_name=self.with_predictor.name()
            )
        return Map2Result(game_map2svm, game_result)
