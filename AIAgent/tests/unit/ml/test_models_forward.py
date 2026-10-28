"""Model-forward smoke tests and ``save_model`` coverage.

The registered models forward a synthetic ``hetero_factory`` graph through
the production ``infer`` entrypoint; ``modelop.filemanager.save_model`` is
exercised with ``tmp_path`` and a patched ``datetime``.
"""

import importlib
from collections import namedtuple

import pytest
import torch
from ml.inference import TORCH, infer
from ml.models.NorthernPenguin.model import StateModelEncoder as NorthernPenguinEncoder
from ml.models.modelop.filemanager import save_model

pytestmark = pytest.mark.unit

GAME_FEATURES_WITH_HISTORY = 7

NORTHERN_PENGUIN_INIT_KWARGS = {
    "hidden_channels": 8,
    "num_of_state_features": 6,
    "num_hops_1": 1,
    "num_hops_2": 1,
    "normalization": True,
    "num_pc_layers": 1,
}

ModelCase = namedtuple(
    "ModelCase",
    ["module", "class_name", "init_kwargs", "game_features", "via_infer", "normalized"],
)

MODEL_CASES = [
    pytest.param(
        ModelCase(
            module="ml.models.NorthernPenguin.model",
            class_name="StateModelEncoder",
            init_kwargs=NORTHERN_PENGUIN_INIT_KWARGS,
            game_features=GAME_FEATURES_WITH_HISTORY,
            via_infer=True,
            normalized=False,
        ),
        id="NorthernPenguin",
    ),
]


def _build_model(case: ModelCase) -> torch.nn.Module:
    module = importlib.import_module(case.module)
    model_class = getattr(module, case.class_name)
    model = model_class(**case.init_kwargs)
    model.eval()
    return model


def _forward(model: torch.nn.Module, data, case: ModelCase) -> torch.Tensor:
    if case.via_infer:
        return infer(model, data)
    raise AssertionError("direct forward is no longer supported")


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

    monkeypatch.setattr(filemanager, "datetime", FixedDatetime)
    monkeypatch.chdir(tmp_path)
    save_dir = tmp_path / "ml" / "models" / "NorthernPenguin"
    save_dir.mkdir(parents=True)
    model = NorthernPenguinEncoder(**NORTHERN_PENGUIN_INIT_KWARGS)

    save_model(model, **NORTHERN_PENGUIN_INIT_KWARGS)

    initargs = "_".join(
        f"{name}_{value}" for name, value in NORTHERN_PENGUIN_INIT_KWARGS.items()
    )
    expected_file = save_dir / (
        f"{type(model).__module__}.{type(model).__name__}_{initargs}_TIMESTAMP.pt"
    )
    assert expected_file.exists()
    loaded = torch.load(expected_file, weights_only=True)
    assert set(loaded) == set(model.state_dict())
