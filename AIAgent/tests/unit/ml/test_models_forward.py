"""Model-forward smoke tests and ``save_model`` coverage.

Every model registered in ``ml.models.registry`` is instantiated with its
small init kwargs and forwards a synthetic ``hetero_factory`` graph through
the production ``infer`` entrypoint — the smoke test is the forward
contract's enforcement: a registered model that cannot forward a dummy
batch fails the suite. ``modelop.filemanager.save_model`` is exercised with
``tmp_path`` and a patched ``datetime``.
"""

import pytest
import torch
from ml.inference import TORCH, infer
from ml.models.modelop.filemanager import save_model
from ml.models.registry import get_model_class, model_names

pytestmark = pytest.mark.unit

# Hyperparameters are test data, not registry identity: the table supplies
# each registered model's small init kwargs. The coverage assertion keeps
# the table and the registry in lockstep.
INIT_KWARGS_BY_NAME = {
    "NorthernPenguin": {
        "hidden_channels": 8,
        "num_of_state_features": 6,
        "num_hops_1": 1,
        "num_hops_2": 1,
        "normalization": True,
        "num_pc_layers": 1,
    },
}


def test_init_kwargs_table_covers_the_registry() -> None:
    assert set(INIT_KWARGS_BY_NAME) == set(model_names())


@pytest.mark.parametrize("name", model_names())
def test_registered_model_forwards_a_dummy_batch(name: str, hetero_factory) -> None:
    model = get_model_class(name)(**INIT_KWARGS_BY_NAME[name])
    model.eval()
    data = hetero_factory()

    output = infer(model, data)

    num_states = data[TORCH.state_vertex].x.shape[0]
    assert output.dtype == torch.float32
    assert output.shape == (num_states, 1)
    assert torch.isfinite(output).all()


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
    init_kwargs = INIT_KWARGS_BY_NAME["NorthernPenguin"]
    save_dir = tmp_path / "ml" / "models" / "NorthernPenguin"
    save_dir.mkdir(parents=True)
    model = get_model_class("NorthernPenguin")(**init_kwargs)

    save_model(model, **init_kwargs)

    initargs = "_".join(f"{name}_{value}" for name, value in init_kwargs.items())
    expected_file = save_dir / (
        f"{type(model).__module__}.{type(model).__name__}_{initargs}_TIMESTAMP.pt"
    )
    assert expected_file.exists()
    loaded = torch.load(expected_file, weights_only=True)
    assert set(loaded) == set(model.state_dict())
