import logging
from pathlib import Path

from common.game import GameMap
from connection.game_server_conn.unsafe_json import asdict
from ml.dataset import get_hetero_data
from ml.validation.coverage.game_managers.each_step.game_states_utils import (
    update_game_state,
)
from ml.validation.coverage.game_managers.model.classes import (
    ModelGameMapInfo,
    ModelGameStep,
)
from torch_geometric.data.hetero_data import HeteroData


def get_steps_from_svm(
    game_map: GameMap, game_map_info: ModelGameMapInfo, game_output_dir: Path
) -> list[ModelGameStep]:
    proc = game_map_info.proc
    if proc is None:
        logging.error("I can't get the steps of a failed game (there was no game)")
        return []
    steps = game_output_dir / f"{game_map.MapName}_steps"
    proc.wait()
    with open(steps, "r") as f:
        steps_serialized = f.read()
    steps = ModelGameStep.schema().loads(steps_serialized, many=True)  # type: ignore
    return steps


def convert_steps_to_hetero(steps: list[ModelGameStep]) -> list[HeteroData]:
    if not steps:
        return []
    step = steps.pop(0)
    game_state, nn_output = step.GameState, step.Output

    steps_hetero = [get_hetero_data(game_state, nn_output)]
    for step in steps:
        delta = step.GameState
        game_state = update_game_state(game_state, delta)
        hetero_input = get_hetero_data(game_state, step.Output)
        steps_hetero.append(hetero_input)
    return steps_hetero
