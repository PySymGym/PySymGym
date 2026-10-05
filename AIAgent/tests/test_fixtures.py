"""Self-tests for the shared AIAgent fixtures (task #558, Phase 0)."""

import pytest
import torch

from ml.inference import TORCH, infer
from ml.models.NorthernPenguin.model import StateModelEncoder


@pytest.mark.unit
def test_hetero_factory_builds_every_node_and_edge_type(hetero_factory):
    data = hetero_factory()

    assert data[TORCH.game_vertex].x.shape == (3, 7)
    assert data[TORCH.state_vertex].x.shape == (2, 6)
    assert data[TORCH.path_condition_vertex].x.shape == (2, 48)

    for edge_type in (
        TORCH.gamevertex_to_gamevertex,
        TORCH.gamevertex_history_statevertex,
        TORCH.gamevertex_in_statevertex,
        TORCH.statevertex_parentof_statevertex,
        TORCH.pathcondvertex_to_pathcondvertex,
        TORCH.pathcondvertex_to_statevertex,
    ):
        assert data[*edge_type].edge_index.shape[0] == 2


@pytest.mark.unit
def test_legacy_game_feature_width_override(hetero_factory):
    data = hetero_factory(game_features=5)

    assert data[TORCH.game_vertex].x.shape == (3, 5)


@pytest.mark.unit
def test_hetero_factory_drives_a_model_forward(hetero_factory):
    data = hetero_factory()
    model = StateModelEncoder(
        hidden_channels=8,
        num_of_state_features=6,
        num_hops_1=1,
        num_hops_2=1,
        normalization=True,
        num_pc_layers=1,
    )

    output = infer(model, data)

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
