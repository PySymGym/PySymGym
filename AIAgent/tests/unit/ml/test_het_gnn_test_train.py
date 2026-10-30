"""Unit tests for ``ml.het_gnn_test_train``.

``train_and_save`` is driven as a seam: a fake model class (one parameter,
deterministic output) and pickled synthetic ``HeteroData`` files, no
end-to-end training run. ``train`` / ``tst`` / ``save_simple`` are also
driven directly against reference math (one Adam step on
``mean((w + 0.5) ** 2)`` from ``w = 0``).
"""

import pickle
import sys
import types
from pathlib import Path

import pytest
import torch
from ml import het_gnn_test_train
from ml.schema import NUM_PC_FEATURES, NUM_STATE_FEATURES
from ml.het_gnn_test_train import HetGNNTestTrain
from ml.inference import TORCH
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

pytestmark = pytest.mark.unit


class FakeHetModel(torch.nn.Module):
    """A model with one parameter and an output that depends on it.

    The output is ``weight.expand(num_states, 8) + 0.5``, so the mse loss
    against zero targets is exactly ``(weight + 0.5) ** 2`` and the gradient
    is ``2 * (weight + 0.5)`` — the reference math the tests assert against.
    """

    calls = 0

    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs) -> torch.Tensor:
        FakeHetModel.calls += 1
        return self.weight.expand(kwargs["state_x"].size(0), 8) + 0.5


# ``get_module_name`` takes ``__module__.split(".")[-2]``; point the fake at
# a deterministic (registered in ``fake_model_module``) module path so the
# checkpoint dir name is stable.
FakeHetModel.__module__ = "ml.models.FakeHet.model"


@pytest.fixture
def fake_model_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the fake model's declared module so its pickles reload."""
    package = types.ModuleType("ml.models.FakeHet")
    module = types.ModuleType("ml.models.FakeHet.model")
    module.FakeHetModel = FakeHetModel
    package.model = module
    monkeypatch.setitem(sys.modules, "ml.models.FakeHet", package)
    monkeypatch.setitem(sys.modules, "ml.models.FakeHet.model", module)


def _reference_adam_step() -> float:
    """One ``Adam(lr=1e-4)`` step on ``mean((w + 0.5) ** 2)`` from ``w = 0``."""
    weight = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([weight], lr=1e-4)
    ((weight + 0.5) ** 2).mean().backward()
    optimizer.step()
    return weight.item()


def _graph_with_targets(hetero_factory) -> HeteroData:
    graph = hetero_factory()
    graph["y"] = torch.zeros(2, 8)
    return graph


def _minimal_graph() -> HeteroData:
    """A one-state graph with every edge type ``infer`` reads, all empty."""
    graph = HeteroData()
    graph[TORCH.game_vertex].x = torch.zeros(1, 7)
    graph[TORCH.state_vertex].x = torch.zeros(1, NUM_STATE_FEATURES)
    graph[TORCH.path_condition_vertex].x = torch.zeros(1, NUM_PC_FEATURES)
    graph[TORCH.gamevertex_to_gamevertex].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    graph[TORCH.gamevertex_to_gamevertex].edge_type = torch.empty(
        (0,), dtype=torch.long
    )
    graph[TORCH.gamevertex_history_statevertex].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    graph[TORCH.gamevertex_history_statevertex].edge_attr = torch.empty(
        (0, 2), dtype=torch.long
    )
    graph[TORCH.gamevertex_in_statevertex].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    graph[TORCH.statevertex_parentof_statevertex].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    graph[TORCH.pathcondvertex_to_pathcondvertex].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    graph[TORCH.pathcondvertex_to_statevertex].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    graph["y"] = torch.zeros(1, 8)
    return graph


def _write_pickle(directory: Path, name: str, graphs: list) -> None:
    with open(directory / name, "wb") as f:
        pickle.dump(graphs, f)


def test_train_and_save_trains_evaluates_and_saves(
    tmp_path: Path,
    hetero_factory,
    capsys: pytest.CaptureFixture[str],
    fake_model_module,
) -> None:
    FakeHetModel.calls = 0
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    graphs = [_graph_with_targets(hetero_factory) for _ in range(12)]
    _write_pickle(dataset_dir, "chunk_0.pkl", graphs[:6])
    _write_pickle(dataset_dir, "chunk_1.pkl", graphs[6:])

    trainer = HetGNNTestTrain(FakeHetModel, hidden=4)
    trainer.train_and_save(dataset_dir, epochs=1, dir_to_save=tmp_path / "save")

    out = capsys.readouterr().out
    # 12 graphs, split at round(12 * 0.85) = 10.
    assert "Number of training graphs: 10" in out
    assert "Number of test graphs: 2" in out
    # Both evals run after the single optimizer step, so both losses are the
    # reference post-step value.
    w1 = _reference_adam_step()
    expected_loss = f"{(w1 + 0.5) ** 2:.6f}"
    assert f"Epoch: 001, Train Loss: {expected_loss}, Test Loss: {expected_loss}" in out

    # 1 train batch (10 graphs, batch_size=100) + 1 train-eval batch +
    # 2 test-eval batches (batch_size=1).
    assert FakeHetModel.calls == 4

    model_dir = tmp_path / "save" / "FakeHet" / "4ch" / "1e"
    state_dict = torch.load(model_dir / "GNN_state_pred_het_dict", weights_only=True)
    assert set(state_dict) == {"weight"}
    assert torch.isclose(state_dict["weight"], torch.tensor(w1))
    loaded = torch.load(model_dir / "GNN_state_pred_het_full", weights_only=False)
    assert isinstance(loaded, FakeHetModel)
    assert torch.isclose(loaded.weight, torch.tensor(w1))


@pytest.mark.slow
def test_train_and_save_balances_oversized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_model_module,
) -> None:
    monkeypatch.setattr(het_gnn_test_train, "BALANCE_DATASET", True)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_pickle(dataset_dir, "big.pkl", [_minimal_graph() for _ in range(5001)])

    trainer = HetGNNTestTrain(FakeHetModel, hidden=4)
    trainer.train_and_save(dataset_dir, epochs=1, dir_to_save=tmp_path / "save")

    out = capsys.readouterr().out
    assert "Part of the dataset is chosen!" in out
    # 5001 truncated to 5000, split at round(5000 * 0.85) = 4250.
    assert "Number of training graphs: 4250" in out
    assert "Number of test graphs: 750" in out
    assert (
        tmp_path / "save" / "FakeHet" / "4ch" / "1e" / "GNN_state_pred_het_dict"
    ).exists()


def test_tst_returns_the_mean_per_state_mse(hetero_factory) -> None:
    trainer = HetGNNTestTrain(FakeHetModel, hidden=4)
    model = FakeHetModel(hidden_channels=4, out_channels=8)
    graphs = [_graph_with_targets(hetero_factory) for _ in range(2)]
    loader = DataLoader(graphs, batch_size=1, shuffle=False)

    loss = trainer.tst(model, loader, torch.device("cpu"))

    # weight = 0: every row is 0.5, the per-row mse against zeros is 0.25.
    assert float(loss) == 0.25


def test_train_performs_one_adam_step_per_batch(hetero_factory) -> None:
    trainer = HetGNNTestTrain(FakeHetModel, hidden=4)
    model = FakeHetModel(hidden_channels=4, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    graphs = [_graph_with_targets(hetero_factory) for _ in range(2)]
    loader = DataLoader(graphs, batch_size=100, shuffle=False)

    trainer.train(model, loader, optimizer, torch.device("cpu"))

    assert torch.isclose(model.weight, torch.tensor(_reference_adam_step()))
    assert model.weight.item() < 0


def test_save_simple_writes_dict_and_full_checkpoints(
    tmp_path: Path, fake_model_module
) -> None:
    trainer = HetGNNTestTrain(FakeHetModel, hidden=4)
    model = FakeHetModel(hidden_channels=4, out_channels=8)
    save_root = tmp_path / "models"

    trainer.save_simple(model, save_root, 3)
    # The second call takes the dir-exists branch (no makedirs).
    trainer.save_simple(model, save_root, 3)

    model_dir = save_root / "FakeHet" / "4ch" / "3e"
    state_dict = torch.load(model_dir / "GNN_state_pred_het_dict", weights_only=True)
    assert set(state_dict) == set(model.state_dict())
    loaded = torch.load(model_dir / "GNN_state_pred_het_full", weights_only=False)
    assert isinstance(loaded, FakeHetModel)
