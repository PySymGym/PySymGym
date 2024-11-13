import json
import os
import typing as t
from pathlib import Path

import pytest
import torch
import yaml
from common.game import GameState
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder as RealStateModelEncoder,
)
from onyx import entrypoint
from tests.utils import read_configs


class TestONNXConversion:
    @pytest.mark.parametrize(
        "config", read_configs("tests/resources/model_configurations")
    )
    def test_onnx_conversion_successful_on_real_randomized_model(
        self, tmp_path, game_states_fixture, config
    ):
        init_game_state: GameState = game_states_fixture[0]
        verification_game_states: list[GameState] = game_states_fixture[1:]

        with open(config) as file:
            model_kwargs = yaml.safe_load(file)

        mock_model_path = tmp_path / "real_random_model.pt"
        torch.save(
            obj=RealStateModelEncoder(**model_kwargs).state_dict(), f=mock_model_path
        )
        mock_onnx_output_path = tmp_path / "real_random_model.onnx"

        entrypoint(
            sample_gamestate=init_game_state,
            pytorch_model_path=mock_model_path,
            onnx_savepath=mock_onnx_output_path,
            model_def=RealStateModelEncoder,
            model_kwargs=model_kwargs,
            verification_gamestates=verification_game_states,
        )

    @pytest.fixture
    def game_states_fixture(request):
        game_states_path = Path("../resources/onnx/reference_gamestates")
        json_files = [
            game_states_path / file
            for file in os.listdir(game_states_path)
            if file.endswith(".json")
        ]
        return [_load_gamestate(it) for it in json_files]


def _load_gamestate(path) -> dict[str, t.Any]:
    with open(path) as f:
        game_state = GameState.from_dict(json.load(f))
        return game_state
