"""Unit tests for the ``ml.training.train`` loop.

The loop is driven with a real small ``NorthernPenguin`` model, a real
``Adam`` optimizer and a real ``BCEWithLogitsLoss(reduction="sum")`` wrapped
in a target-capturing recorder, so the pinned behaviour is the production
one: the ``PYTORCH_CUDA_ALLOC_CONF`` switch, the ``y_true`` clamping to
``{0, 1}`` before the criterion, and the per-batch optimizer step. The
zero-loss branch is driven with a fake criterion and a spy optimizer.
"""

import os

import pytest
import torch
import torch.nn as nn
from ml.models.NorthernPenguin.model import StateModelEncoder as NorthernPenguinEncoder
from ml.training.train import train
from torch_geometric.loader import DataLoader

pytestmark = pytest.mark.unit

SMALL_MODEL_KWARGS = {
    "hidden_channels": 8,
    "num_of_state_features": 6,
    "num_hops_1": 1,
    "num_hops_2": 1,
    "normalization": True,
    "num_pc_layers": 1,
}


class FixedOutputModel(torch.nn.Module):
    """A model with one parameter and a fixed, input-independent output."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs) -> torch.Tensor:
        return torch.zeros(2, 1)


class SpyOptimizer(torch.optim.Optimizer):
    """An optimizer that records ``step`` calls without updating parameters."""

    def __init__(self, params) -> None:
        super().__init__(params, {"lr": 1e-3})
        self.step_calls = 0

    def step(self, closure=None) -> None:
        self.step_calls += 1


def test_train_sets_cuda_alloc_conf_clamps_targets_and_updates_weights(
    monkeypatch: pytest.MonkeyPatch, step_factory
) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    real_criterion = nn.BCEWithLogitsLoss(reduction="sum")
    captured_targets: list[torch.Tensor] = []

    def criterion(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        captured_targets.append(target.clone())
        return real_criterion(out, target)

    steps = [
        step_factory(torch.tensor([[0.0], [5.0]])),
        step_factory(torch.tensor([[7.0], [0.0]])),
    ]
    weights_before = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }

    train(DataLoader(steps, batch_size=1), model, optimizer, criterion)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    # Every positive target value is clamped to 1 before the criterion sees it.
    assert [t.tolist() for t in captured_targets] == [
        [[0.0], [1.0]],
        [[1.0], [0.0]],
    ]
    changed = [
        name
        for name, param in model.state_dict().items()
        if not torch.equal(param, weights_before[name])
    ]
    assert changed, "the optimizer step must update at least one parameter"


def test_train_skips_backward_and_step_when_the_criterion_returns_zero(
    step_factory,
) -> None:
    model = FixedOutputModel()
    optimizer = SpyOptimizer(model.parameters())
    weight_before = model.weight.detach().clone()

    train(
        DataLoader([step_factory(torch.tensor([[0.0], [1.0]]))], batch_size=1),
        model,
        optimizer,
        lambda out, target: torch.tensor(0.0),
    )

    assert optimizer.step_calls == 0
    assert torch.equal(model.weight, weight_before)
