"""Unit tests for the ONNX naming contract in ``ml.inference``.

``infer`` itself is a thin wrapper already exercised by the fixture self-test
in ``AIAgent/tests/test_fixtures.py``; here we pin the pure ``underscore_join``
helper and the ONNX key derivation it drives.
"""

import pytest
from ml.inference import ONNX, TORCH, underscore_join

pytestmark = pytest.mark.unit


def test_underscore_join_strips_underscores_then_joins():
    assert underscore_join(["a_b", "c_d"]) == "ab_cd"
    assert (
        underscore_join(("game_vertex", "to", "game_vertex"))
        == "gamevertex_to_gamevertex"
    )


def test_underscore_join_always_separates_with_an_underscore():
    assert (
        underscore_join(("gamevertex", "to", "gamevertex"))
        == "gamevertex_to_gamevertex"
    )


def test_onnx_edge_keys_are_underscore_joined_torch_tuples():
    assert ONNX.gamevertex_to_gamevertex_index == underscore_join(
        TORCH.gamevertex_to_gamevertex + ("index",)
    )
    assert ONNX.gamevertex_to_gamevertex_type == underscore_join(
        TORCH.gamevertex_to_gamevertex + ("type",)
    )
    assert ONNX.gamevertex_history_statevertex_attrs == underscore_join(
        TORCH.gamevertex_history_statevertex + ("attrs",)
    )


def test_onnx_vertex_keys_match_torch_vertex_keys():
    assert ONNX.game_vertex == TORCH.game_vertex
    assert ONNX.state_vertex == TORCH.state_vertex
    assert ONNX.path_condition_vertex == TORCH.path_condition_vertex
