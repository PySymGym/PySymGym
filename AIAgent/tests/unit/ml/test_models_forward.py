"""Model-forward smoke tests and ``save_model`` coverage.

Only the models that forward unchanged against the production tensor schema
are exercised here: ``NorthernPenguin`` and ``InvisibleCow`` (both game width
7, six state features). The legacy architectures under ``ml/models/`` cannot
consume the current graph tensors and are tracked in #571/#572 instead.

``NorthernPenguin`` goes through the production ``infer`` entrypoint;
``InvisibleCow`` needs the extra reverse/path-condition edges, so it is called
with the subset of inputs its ``forward`` declares.
"""

import importlib
import inspect
from collections import namedtuple

import pytest
import torch
from ml.inference import TORCH, infer
from ml.models.NorthernPenguin.model import StateModelEncoder as NorthernPenguinEncoder
from ml.models.modelop.filemanager import save_model

pytestmark = pytest.mark.unit

GAME_FEATURES_WITH_HISTORY = 7

ModelCase = namedtuple(
    "ModelCase",
    ["module", "class_name", "init_kwargs", "game_features", "via_infer", "normalized"],
)

MODEL_CASES = [
    pytest.param(
        ModelCase(
            module="ml.models.NorthernPenguin.model",
            class_name="StateModelEncoder",
            init_kwargs={
                "hidden_channels": 8,
                "num_of_state_features": 6,
                "num_hops_1": 1,
                "num_hops_2": 1,
                "normalization": True,
                "num_pc_layers": 1,
            },
            game_features=GAME_FEATURES_WITH_HISTORY,
            via_infer=True,
            normalized=False,
        ),
        id="NorthernPenguin",
    ),
    pytest.param(
        ModelCase(
            module="ml.models.InvisibleCow.model",
            class_name="StateModelEncoder",
            init_kwargs={
                "hidden_channels": 8,
                "num_of_state_features": 6,
                "blocks": [8, 8],
                "normalization": True,
            },
            game_features=GAME_FEATURES_WITH_HISTORY,
            via_infer=False,
            normalized=True,
        ),
        id="InvisibleCow",
    ),
]


def _model_inputs(data) -> dict:
    return {
        "game_x": data[TORCH.game_vertex].x,
        "state_x": data[TORCH.state_vertex].x,
        "pc_x": data[TORCH.path_condition_vertex].x,
        "edge_index_v_v": data[*TORCH.gamevertex_to_gamevertex].edge_index,
        "edge_type_v_v": data[*TORCH.gamevertex_to_gamevertex].edge_type,
        "edge_index_history_v_s": data[
            *TORCH.gamevertex_history_statevertex
        ].edge_index,
        "edge_attr_history_v_s": data[*TORCH.gamevertex_history_statevertex].edge_attr,
        "edge_index_history_s_v": data[
            *TORCH.statevertex_history_gamevertex
        ].edge_index,
        "edge_index_in_v_s": data[*TORCH.gamevertex_in_statevertex].edge_index,
        "edge_index_in_s_v": data[*TORCH.statevertex_in_gamevertex].edge_index,
        "edge_index_s_s": data[*TORCH.statevertex_parentof_statevertex].edge_index,
        "edge_index_pc_pc": data[*TORCH.pathcondvertex_to_pathcondvertex].edge_index,
        "edge_index_pc_s": data[*TORCH.pathcondvertex_to_statevertex].edge_index,
        "edge_index_s_pc": data[*TORCH.statevertex_to_pathcondvertex].edge_index,
    }


def _build_model(case: ModelCase) -> torch.nn.Module:
    module = importlib.import_module(case.module)
    model_class = getattr(module, case.class_name)
    model = model_class(**case.init_kwargs)
    model.eval()
    return model


def _forward(model: torch.nn.Module, data, case: ModelCase) -> torch.Tensor:
    if case.via_infer:
        return infer(model, data)
    parameters = inspect.signature(model.forward).parameters
    inputs = {
        name: value for name, value in _model_inputs(data).items() if name in parameters
    }
    return model(**inputs)


@pytest.mark.parametrize("case", MODEL_CASES)
def test_model_forward_smoke(case: ModelCase, hetero_factory) -> None:
    data = hetero_factory(game_features=case.game_features)
    model = _build_model(case)

    output = _forward(model, data, case)

    num_states = data[TORCH.state_vertex].x.shape[0]
    assert output.dtype == torch.float32
    assert output.shape == (num_states, 1)
    assert torch.isfinite(output).all()
    if case.normalized:
        assert torch.isclose(torch.exp(output).sum(), torch.tensor(1.0), atol=1e-5)


def test_save_model_writes_a_reloadable_state_dict(monkeypatch, tmp_path) -> None:
    import ml.models.modelop.filemanager as filemanager

    class FixedDatetime:
        @classmethod
        def now(cls):
            return cls()

        def timestamp(self) -> float:
            return 0.0

        @classmethod
        def fromtimestamp(cls, timestamp: float) -> str:
            return "TIMESTAMP"

    init_kwargs = {
        "hidden_channels": 8,
        "num_of_state_features": 6,
        "num_hops_1": 1,
        "num_hops_2": 1,
        "normalization": True,
        "num_pc_layers": 1,
    }
    monkeypatch.setattr(filemanager, "datetime", FixedDatetime)
    monkeypatch.chdir(tmp_path)
    save_dir = tmp_path / "ml" / "models" / "NorthernPenguin"
    save_dir.mkdir(parents=True)
    model = NorthernPenguinEncoder(**init_kwargs)

    save_model(model, **init_kwargs)

    initargs = "_".join(f"{name}_{value}" for name, value in init_kwargs.items())
    expected_file = save_dir / (
        f"{type(model).__module__}.{type(model).__name__}_{initargs}_TIMESTAMP.pt"
    )
    assert expected_file.exists()
    loaded = torch.load(expected_file, weights_only=True)
    assert set(loaded) == set(model.state_dict())
