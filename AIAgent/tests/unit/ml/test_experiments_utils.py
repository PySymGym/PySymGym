"""Unit tests for the pure helpers in ``ml.training.experiments_utils``."""

import pytest
import torch
from ml.training.experiments_utils import (
    euclidean_dist,
    find_entry_points,
    remove_call_return_edges,
)
from torch_geometric.data import Data

pytestmark = pytest.mark.unit


def test_euclidean_dist_is_zero_for_a_single_point():
    assert euclidean_dist(torch.tensor([[1.0]]), torch.tensor([[0.0]])) == 0


def test_euclidean_dist_min_shifts_both_tensors():
    y_pred = torch.tensor([[0.0], [2.0]])
    y_true = torch.tensor([[0.0], [0.0]])

    assert torch.isclose(euclidean_dist(y_pred, y_true), torch.tensor(2.0))


def test_euclidean_dist_is_translation_invariant():
    y_pred = torch.tensor([[0.0], [3.0]])
    y_true = torch.tensor([[1.0], [2.0]])
    shift = 5.0

    assert torch.isclose(
        euclidean_dist(y_pred + shift, y_true + shift),
        euclidean_dist(y_pred, y_true),
    )


def test_remove_call_return_edges_drops_types_one_and_two():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    cfg = Data(
        x=torch.zeros(4, 1),
        edge_index=edge_index,
        edge_attr=torch.tensor([0, 1, 2, 0]),
    )

    result = remove_call_return_edges(cfg)

    assert torch.equal(result.edge_index, torch.tensor([[0, 3], [1, 0]]))


def test_remove_call_return_edges_keeps_everything_without_call_edges():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    cfg = Data(
        x=torch.zeros(2, 1),
        edge_index=edge_index.clone(),
        edge_attr=torch.tensor([0, 0]),
    )

    result = remove_call_return_edges(cfg)

    assert torch.equal(result.edge_index, edge_index)


def test_remove_call_return_edges_handles_empty_graph():
    cfg = Data(
        x=torch.zeros(2, 1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0,), dtype=torch.long),
    )

    result = remove_call_return_edges(cfg)

    assert result.edge_index.shape == (2, 0)


def test_find_entry_points_returns_vertices_that_are_never_a_target():
    cfg = Data(
        x=torch.zeros(4, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )

    assert find_entry_points(cfg) == [0, 3]


def test_find_entry_points_returns_all_vertices_for_an_empty_graph():
    cfg = Data(
        x=torch.zeros(3, 1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )

    assert find_entry_points(cfg) == [0, 1, 2]
