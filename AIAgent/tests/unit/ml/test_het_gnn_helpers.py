"""Unit tests for the pure helpers in ``ml.het_gnn_test_train``."""

import pytest
import torch
import torch.nn.functional as F
from ml.het_gnn_test_train import HetGNNTestTrain, get_module_name
from ml.models.InvisibleCow.model import StateModelEncoder as InvisibleCowEncoder
from ml.models.NorthernPenguin.model import StateModelEncoder as NorthernPenguinEncoder

pytestmark = pytest.mark.unit


@pytest.fixture
def trainer() -> HetGNNTestTrain:
    return HetGNNTestTrain(model_class=None, hidden=1)


def test_get_module_name_returns_the_parent_package():
    assert get_module_name(NorthernPenguinEncoder) == "NorthernPenguin"
    assert get_module_name(InvisibleCowEncoder) == "InvisibleCow"


def test_weighted_mse_loss_without_weight_equals_mse(trainer):
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 1.0, 1.0])

    assert torch.isclose(
        trainer.weighted_mse_loss(pred, target), F.mse_loss(pred, target)
    )


def test_weighted_mse_loss_indexes_the_weight_by_target(trainer):
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([0, 1])
    weight = torch.tensor([1.0, 3.0])

    assert torch.isclose(
        trainer.weighted_mse_loss(pred, target, weight), torch.tensor(2.0)
    )


def test_weighted_mse_loss_casts_the_weight_to_the_prediction_dtype(trainer):
    pred = torch.zeros(2, dtype=torch.float32)
    target = torch.tensor([0, 1])
    weight = torch.tensor([1.0, 1.0], dtype=torch.float64)

    assert trainer.weighted_mse_loss(pred, target, weight).dtype == torch.float32
