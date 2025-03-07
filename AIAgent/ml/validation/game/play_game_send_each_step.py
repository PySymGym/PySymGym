import logging
import traceback
from common.classes import GameFailed, GameResult, Map2Result
from common.game import GameState, GameMap2SVM
from config import FeatureConfig
from connection.broker_conn.socket_manager import game_server_socket_manager
from connection.errors_connection import GameInterruptedError
from connection.game_server_conn.connector import Connector
from func_timeout import FunctionTimedOut, func_set_timeout
from ml.protocols import Predictor
from ml.dataset import Result, TrainingDataset, convert_input_to_tensor
from time import perf_counter
from typing import TypeAlias

TimeDuration: TypeAlias = float


def get_states(game_state: GameState) -> set[int]:
    return {s.Id for s in game_state.States}


def update_game_state(game_state: GameState, delta: GameState) -> GameState:
    if game_state is None:
        return delta

    updated_basic_blocks = {v.Id for v in delta.GraphVertices}
    updated_states = {s.Id for s in delta.States}

    vertices = [
        v for v in game_state.GraphVertices if v.Id not in updated_basic_blocks
    ] + delta.GraphVertices

    edges = [
        e for e in game_state.Map if e.VertexFrom not in updated_basic_blocks
    ] + delta.Map

    active_states = {state for v in vertices for state in v.States}
    new_states = [
        s
        for s in game_state.States
        if s.Id in active_states and s.Id not in updated_states
    ] + delta.States
    for s in new_states:
        s.Children = list(filter(lambda c: c in active_states, s.Children))

    return GameState(vertices, new_states, edges)


def play_map(
    with_connector: Connector, with_predictor: Predictor, with_dataset: TrainingDataset
) -> tuple[GameResult, TimeDuration]:
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

            predicted_state_id, nn_output = with_predictor.predict(game_state)

            add_single_step(game_state, nn_output)

            logging.debug(
                f"<{with_predictor.name()}> step: {steps_count}, available states: {get_states(game_state)}, predicted: {predicted_state_id}"
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
                f"<{with_predictor.name()}>: immediate GameOver on {with_connector.map.MapName}"
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
            f"<{with_predictor.name()}>: not all steps exshausted on {with_connector.map.MapName} with non-100% coverage"
            f"steps taken: {steps_count}, actual coverage: {actual_coverage:.2f}"
        )
        steps_count = steps

    model_result = GameResult(
        steps_count=steps_count,
        tests_count=tests_count,
        errors_count=errors_count,
        actual_coverage_percent=actual_coverage,
    )
    if with_dataset is not None:
        map_result = Result(
            coverage_percent=model_result.actual_coverage_percent,
            negative_tests_number=-model_result.tests_count,
            negative_steps_number=-model_result.steps_count,
            errors_number=model_result.errors_count,
        )
        with_dataset.update_map(with_connector.map.MapName, map_result, map_steps)
    del map_steps
    return model_result, end_time - start_time


@func_set_timeout(FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.timeout_sec)
def play_map_with_timeout(
    with_connector: Connector, with_predictor: Predictor, with_dataset
) -> tuple[GameResult, TimeDuration]:
    return play_map(with_connector, with_predictor, with_dataset)


def play_game(
    with_predictor: Predictor,
    game_map2svm: GameMap2SVM,
    with_dataset: TrainingDataset,
):
    logging.info(f"<{with_predictor.name()}> is playing {game_map2svm.GameMap.MapName}")
    need_to_save = False
    try:
        play_func = (
            play_map_with_timeout
            if FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.enabled
            else play_map
        )
        with game_server_socket_manager(game_map2svm.SVMInfo) as ws:
            game_result, time = play_func(
                with_connector=Connector(ws, game_map2svm.GameMap),
                with_predictor=with_predictor,
                with_dataset=with_dataset,
            )
        logging.info(
            f"<{with_predictor.name()}> finished map {game_map2svm.GameMap.MapName} "
            f"in {game_result.steps_count} steps, {time} seconds, "
            f"actual coverage: {game_result.actual_coverage_percent:.2f}"
        )
    except FunctionTimedOut as error:
        need_to_save = True
        logging.warning(
            f"<{with_predictor.name()}> timeouted on map {game_map2svm.GameMap.MapName} with {error.timedOutAfter}s"
        )
        game_result = GameFailed(reason=type(error))
    except GameInterruptedError as error:
        logging.warning(
            f"<{with_predictor.name()}> failed on map {game_map2svm.GameMap.MapName} with {error.__class__.__name__}: {error.desc}"
        )
        game_result = GameFailed(reason=type(error))
    except Exception as error:
        need_to_save = True
        logging.warning(
            (
                f"<{with_predictor.name()}> failed on map {game_map2svm.GameMap.MapName}:\n"
                + "\n".join(
                    traceback.format_exception(
                        type(error), value=error, tb=error.__traceback__
                    )
                )
            )
        )
        game_result = GameFailed(reason=type(error))

    map2result = Map2Result(game_map2svm, game_result)
    if need_to_save:
        FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.save_model(
            with_predictor.model(), with_name=with_predictor.name()
        )
    return map2result
