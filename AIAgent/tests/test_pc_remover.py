import numpy as np
import pytest
import torch
from pathlib import Path
from ml.inference import TORCH
from ml.pc_remover import remove_path_condition_root
from tests.resources.heterodata_expected import (
    expected1,
    expected2,
    expected3,
    expected4,
)


@pytest.fixture
def test_data():
    heterodata1 = torch.load(
        Path("tests/resources/heterodata_with_root1.pt"), weights_only=False
    )
    heterodata2 = torch.load(
        Path("tests/resources/heterodata_with_root2.pt"), weights_only=False
    )
    heterodata3 = torch.load(
        Path("tests/resources/heterodata_with_root3.pt"), weights_only=False
    )
    heterodata4 = torch.load(
        Path("tests/resources/heterodata_with_root4.pt"), weights_only=False
    )

    return [
        (remove_path_condition_root(heterodata1), expected1),
        (remove_path_condition_root(heterodata2), expected2),
        (remove_path_condition_root(heterodata3), expected3),
        (remove_path_condition_root(heterodata4), expected4),
    ]


def test_compare_all(test_data):
    for result, expected in test_data:
        assert torch.equal(result[TORCH.game_vertex].x, expected[TORCH.game_vertex].x)
        assert torch.equal(result[TORCH.state_vertex].x, expected[TORCH.state_vertex].x)
        assert torch.equal(
            result[TORCH.path_condition_vertex].x,
            expected[TORCH.path_condition_vertex].x,
        )
        assert torch.equal(
            result[TORCH.gamevertex_to_gamevertex].edge_index,
            expected[TORCH.gamevertex_to_gamevertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.gamevertex_to_gamevertex].edge_type,
            expected[TORCH.gamevertex_to_gamevertex].edge_type,
        )
        assert torch.equal(
            result[TORCH.statevertex_in_gamevertex].edge_index,
            expected[TORCH.statevertex_in_gamevertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.gamevertex_in_statevertex].edge_index,
            expected[TORCH.gamevertex_in_statevertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.statevertex_history_gamevertex].edge_index,
            expected[TORCH.statevertex_history_gamevertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.statevertex_history_gamevertex].edge_attr,
            expected[TORCH.statevertex_history_gamevertex].edge_attr,
        )
        assert torch.equal(
            result[TORCH.gamevertex_history_statevertex].edge_index,
            expected[TORCH.gamevertex_history_statevertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.gamevertex_history_statevertex].edge_attr,
            expected[TORCH.gamevertex_history_statevertex].edge_attr,
        )
        assert torch.equal(
            result[TORCH.statevertex_parentof_statevertex].edge_index,
            expected[TORCH.statevertex_parentof_statevertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.pathcondvertex_to_pathcondvertex].edge_index,
            expected[TORCH.pathcondvertex_to_pathcondvertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.pathcondvertex_to_statevertex].edge_index,
            expected[TORCH.pathcondvertex_to_statevertex].edge_index,
        )
        assert torch.equal(
            result[TORCH.statevertex_to_pathcondvertex].edge_index,
            expected[TORCH.statevertex_to_pathcondvertex].edge_index,
        )


def test_non_all_features_are_zeros(test_data):
    for result, _ in test_data:
        for vector in result[TORCH.game_vertex].x.detach().cpu().numpy():
            assert not np.all(vector == 0)
