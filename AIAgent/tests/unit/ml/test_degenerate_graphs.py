"""Regression tests for degenerate graphs (#619).

A degenerate step converts to a graph with no states or no game vertices.
Before the fix, the empty node lists became 1-D ``(0,)`` tensors and the
model forward crashed (``gcn_conv`` ``IndexError``); the training loop then
crashed at ``loss.backward()`` (the guard's constant output has no
``grad_fn`` while the loss is non-zero for a zero-vertices step). Zero
path-condition vertices are NOT degenerate: a state with no path condition
is valid and the model handles it.
"""

import logging

import pytest
import torch
import torch.nn as nn
from ml.dataset import (
    NUM_CFG_VERTEX_FEATURES,
    NUM_PC_FEATURES,
    NUM_STATE_FEATURES,
    convert_input_to_tensor,
)
from ml.inference import TORCH, degenerate_counts, infer
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


class _UncallableModel:
    """A stand-in that fails loudly if a degenerate graph reaches it."""

    def __call__(self, **kwargs) -> torch.Tensor:
        raise AssertionError("the model must not be called on a degenerate graph")


def _zero_pc(gamestate_factory):
    """A valid step whose states carry no path condition."""
    gs = gamestate_factory()
    gs.PathConditionVertices = []
    for state in gs.States:
        state.PathCondition = []
    return gs


def _zero_states(gamestate_factory):
    """A degenerate step with no states (and no vertex->state references)."""
    gs = gamestate_factory()
    gs.States = []
    for vertex in gs.GraphVertices:
        vertex.States = []
    return gs


def _zero_vertices(gamestate_factory):
    """A degenerate step with no game vertices (and no references to them)."""
    gs = gamestate_factory()
    gs.GraphVertices = []
    gs.Map = []
    for state in gs.States:
        state.History = []
    return gs


NODE_SHAPES = {
    TORCH.game_vertex: (3, NUM_CFG_VERTEX_FEATURES),
    TORCH.state_vertex: (2, NUM_STATE_FEATURES),
    TORCH.path_condition_vertex: (2, NUM_PC_FEATURES),
}


@pytest.mark.parametrize(
    ("mutate", "empty_type"),
    [
        pytest.param(_zero_pc, TORCH.path_condition_vertex, id="zero-pc"),
        pytest.param(_zero_states, TORCH.state_vertex, id="zero-states"),
        pytest.param(_zero_vertices, TORCH.game_vertex, id="zero-vertices"),
    ],
)
def test_convert_input_to_tensor_emits_2d_empty_tensors(
    gamestate_factory, mutate, empty_type
):
    graph, _ = convert_input_to_tensor(mutate(gamestate_factory))

    for node_type, (count, width) in NODE_SHAPES.items():
        expected_count = 0 if node_type == empty_type else count
        assert graph[node_type].x.shape == (expected_count, width), node_type


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(_zero_states, (0, 3), id="zero-states"),
        pytest.param(_zero_vertices, (2, 0), id="zero-vertices"),
    ],
)
def test_degenerate_counts_reports_both_counts(gamestate_factory, mutate, expected):
    graph, _ = convert_input_to_tensor(mutate(gamestate_factory))

    assert degenerate_counts(graph) == expected


def test_degenerate_counts_accepts_a_zero_pc_graph(gamestate_factory):
    graph, _ = convert_input_to_tensor(_zero_pc(gamestate_factory))

    assert degenerate_counts(graph) is None


def test_degenerate_counts_accepts_the_normal_graph(gamestate_factory):
    graph, _ = convert_input_to_tensor(gamestate_factory())

    assert degenerate_counts(graph) is None


def test_infer_zero_states_returns_empty_without_calling_the_model(gamestate_factory):
    graph, _ = convert_input_to_tensor(_zero_states(gamestate_factory))

    out = infer(_UncallableModel(), graph)

    assert out.shape == (0, 1)
    assert out.dtype == torch.float32


def test_infer_zero_vertices_returns_neutral_without_calling_the_model(
    gamestate_factory,
):
    graph, _ = convert_input_to_tensor(_zero_vertices(gamestate_factory))
    num_states = graph[TORCH.state_vertex].x.shape[0]

    out = infer(_UncallableModel(), graph)

    assert out.shape == (num_states, 1)
    assert torch.equal(out, torch.zeros((num_states, 1)))


def test_infer_logs_a_warning_with_both_counts(gamestate_factory, caplog):
    graph, _ = convert_input_to_tensor(_zero_states(gamestate_factory))

    with caplog.at_level(logging.WARNING):
        infer(_UncallableModel(), graph)

    assert caplog.messages == ["degenerate graph: 0 states, 3 game vertices"]


def test_infer_zero_pc_forwards_through_the_real_model(gamestate_factory):
    """The exact #619 crash input: a valid step with no path condition."""
    graph, _ = convert_input_to_tensor(_zero_pc(gamestate_factory))
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS).eval()
    num_states = graph[TORCH.state_vertex].x.shape[0]

    out = infer(model, graph)

    assert out.shape == (num_states, 1)
    assert torch.isfinite(out).all()


def _degenerate_step(gamestate_factory, mutate, y_true):
    graph, _ = convert_input_to_tensor(mutate(gamestate_factory))
    graph.y_true = y_true
    return graph


def test_train_over_a_zero_states_step_is_a_no_op(gamestate_factory):
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    step = _degenerate_step(gamestate_factory, _zero_states, torch.zeros((0, 1)))
    weights_before = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }

    train(DataLoader([step], batch_size=1), model, optimizer, criterion)

    assert all(
        torch.equal(param, weights_before[name])
        for name, param in model.state_dict().items()
    )


def test_train_over_a_zero_vertices_step_is_a_no_op(gamestate_factory):
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    y_true = torch.zeros((2, 1))
    y_true[0] = 1
    step = _degenerate_step(gamestate_factory, _zero_vertices, y_true)
    weights_before = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }

    train(DataLoader([step], batch_size=1), model, optimizer, criterion)

    assert all(
        torch.equal(param, weights_before[name])
        for name, param in model.state_dict().items()
    )


def test_train_over_a_mixed_dataloader_trains_the_normal_step(
    gamestate_factory, step_factory
):
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    normal = step_factory(torch.tensor([[0.0], [1.0]]))
    degenerate = _degenerate_step(gamestate_factory, _zero_states, torch.zeros((0, 1)))
    weights_before = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }

    train(DataLoader([normal, degenerate], batch_size=1), model, optimizer, criterion)

    changed = [
        name
        for name, param in model.state_dict().items()
        if not torch.equal(param, weights_before[name])
    ]
    assert changed, "the normal step must still train"
