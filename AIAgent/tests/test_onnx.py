import copy
import json
import typing as t
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from common.game import GameState
from ml.dataset import convert_input_to_tensor
from ml.inference import TORCH
from ml.models.NorthernPenguin.model import (
    StateModelEncoder as RealStateModelEncoder,
)
from ml.validation.coverage.game_managers.model import process_game_manager
from onyx import create_torch_input, entrypoint, save_in_onnx
from tests_utils import read_configs

pytestmark = pytest.mark.integration

TESTS_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS_DIR = TESTS_DIR / "resources" / "model_configurations"
REFERENCE_GAMESTATES_DIR = (
    TESTS_DIR.parents[1] / "resources" / "onnx" / "reference_gamestates"
)


class TestONNXConversion:
    @pytest.mark.parametrize("config", read_configs(MODEL_CONFIGS_DIR))
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
    def game_states_fixture(self):
        json_files = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))
        return [_load_gamestate(path) for path in json_files]


class TestONNXExportWarnings:
    def test_export_emits_no_warnings(self, tmp_path):
        """save_in_onnx must not emit warnings from torch.onnx, PyG conv, or
        this repo (regression guard for #617)."""
        with open(read_configs(MODEL_CONFIGS_DIR)[0]) as file:
            model_kwargs = yaml.safe_load(file)

        first = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]
        game_state = _load_gamestate(first)

        torch_model = RealStateModelEncoder(**model_kwargs).eval()
        hetero, _ = convert_input_to_tensor(game_state)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_in_onnx(torch_model, hetero, tmp_path / "model.onnx")

        offenders = [
            w
            for w in caught
            if (
                "torch/onnx/" in (w.filename or "")
                or "torch_geometric/nn/conv/" in (w.filename or "")
                or "/AIAgent/" in (w.filename or "")
            )
        ]
        assert not offenders, [
            f"{w.category.__name__} @ {w.filename}:{w.lineno}: {w.message}"
            for w in offenders
        ]


class TestForwardOnCUDA:
    """Regression guard for #617: StateModelEncoder.forward must run on CUDA.
    The v-v skip condition is a Python-int-derived tensor; if it is created
    without an explicit device it lands on CPU and torch.where raises
    "Expected all tensors to be on the same device" for CUDA inputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_forward_matches_cpu_with_and_without_v_v_edges(self):
        with open(read_configs(MODEL_CONFIGS_DIR)[0]) as file:
            model_kwargs = yaml.safe_load(file)

        first = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]
        game_state = _load_gamestate(first)
        hetero, _ = convert_input_to_tensor(game_state)

        empty_hetero = copy.deepcopy(hetero)
        empty_hetero[TORCH.gamevertex_to_gamevertex].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        empty_hetero[TORCH.gamevertex_to_gamevertex].edge_type = torch.empty(
            (0,), dtype=torch.long
        )

        cpu_model = RealStateModelEncoder(**model_kwargs).eval()
        cuda_model = copy.deepcopy(cpu_model).cuda()

        for sample in (hetero, empty_hetero):
            args = create_torch_input(sample)
            expected = cpu_model(*args)
            cuda_args = tuple(arg.to("cuda") for arg in args)
            actual = cuda_model(*cuda_args)
            assert torch.allclose(expected, actual.cpu(), atol=1e-4)


class TestModelGamePreparatorKwargs:
    def test_create_onnx_model_uses_explicit_kwargs(self, tmp_path, monkeypatch):
        with open(read_configs(MODEL_CONFIGS_DIR)[0]) as file:
            model_kwargs = yaml.safe_load(file)

        captured = {}

        def fake_export(
            sample_gamestate,
            pytorch_model_path,
            onnx_savepath,
            model_def,
            model_kwargs,
            verification_gamestates=None,
        ):
            captured["model_kwargs"] = model_kwargs
            captured["pytorch_model_path"] = pytorch_model_path
            captured["onnx_savepath"] = onnx_savepath

        monkeypatch.setattr(
            process_game_manager, "save_torch_model_to_onnx_file", fake_export
        )
        model_path = tmp_path / "model.pth"
        onnx_path = tmp_path / "model.onnx"
        namespace = SimpleNamespace(
            shared_lock=None, is_prepared=SimpleNamespace(value=False)
        )
        preparator = process_game_manager.ModelGamePreparator(
            namespace,
            RealStateModelEncoder(**model_kwargs),
            model_path,
            onnx_path,
            model_kwargs,
        )
        preparator._create_onnx_model()

        assert captured["model_kwargs"] == model_kwargs
        assert captured["pytorch_model_path"] == model_path
        assert captured["onnx_savepath"] == onnx_path


def _load_gamestate(path) -> dict[str, t.Any]:
    with open(path) as f:
        game_state = GameState.from_dict(json.load(f))
        return game_state
