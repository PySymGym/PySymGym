"""Unit tests for the pure helpers of the ONNX export entrypoint ``onyx.py``.

These pin the current (pre-#578) behaviour of the tensor-input builders and
the gamestate loader. The ONNX input names are pinned as literal strings:
they are the V# wire contract and the drift net for #597's single-source
input-names change. Model selection goes through the registry (pinned in
``test_registry.py``).
"""

import json

import numpy as np
import onyx
import pytest
import torch
from common.game import GameState
from ml.inference import TORCH
from onyx import (
    create_model_input,
    create_onnxruntime_input,
    create_torch_input,
    dynamic_axes,
    input_names,
    load_gamestate,
    verify_outputs,
)
from paths import RESOURCES_PATH

pytestmark = pytest.mark.unit

REFERENCE_GAMESTATES_DIR = RESOURCES_PATH / "onnx" / "reference_gamestates"

# The 11 ONNX inputs in the exact ``input_names`` order: the literal wire
# name (the V# contract), the HeteroData edge/node type, and the attribute.
EXPECTED_INPUTS = [
    ("game_vertex", TORCH.game_vertex, "x"),
    ("state_vertex", TORCH.state_vertex, "x"),
    ("path_condition_vertex", TORCH.path_condition_vertex, "x"),
    ("gamevertex_to_gamevertex_index", TORCH.gamevertex_to_gamevertex, "edge_index"),
    ("gamevertex_to_gamevertex_type", TORCH.gamevertex_to_gamevertex, "edge_type"),
    (
        "gamevertex_history_statevertex_index",
        TORCH.gamevertex_history_statevertex,
        "edge_index",
    ),
    (
        "gamevertex_history_statevertex_attrs",
        TORCH.gamevertex_history_statevertex,
        "edge_attr",
    ),
    ("gamevertex_in_statevertex", TORCH.gamevertex_in_statevertex, "edge_index"),
    (
        "statevertex_parentof_statevertex",
        TORCH.statevertex_parentof_statevertex,
        "edge_index",
    ),
    # ``underscore_join`` strips the underscores of the node-type names, so
    # the path-condition wire names differ from the constant names.
    (
        "pathconditionvertex_to_pathconditionvertex",
        TORCH.pathcondvertex_to_pathcondvertex,
        "edge_index",
    ),
    (
        "pathconditionvertex_to_statevertex",
        TORCH.pathcondvertex_to_statevertex,
        "edge_index",
    ),
]


class TestCreateModelInput:
    def test_maps_every_tensor_to_its_onnx_wire_name(self, hetero_factory):
        hetero = hetero_factory()

        result = create_model_input(hetero)

        assert list(result) == [name for name, _, _ in EXPECTED_INPUTS]
        for name, key, attr in EXPECTED_INPUTS:
            assert result[name] is getattr(hetero[key], attr)

    def test_applies_the_modifier_to_every_value(self, hetero_factory):
        hetero = hetero_factory()

        doubled = create_model_input(hetero, lambda x: x * 2)

        for name, key, attr in EXPECTED_INPUTS:
            assert torch.equal(doubled[name], getattr(hetero[key], attr) * 2)


class TestCreateOnnxruntimeInput:
    def test_returns_the_numpy_arrays(self, hetero_factory):
        hetero = hetero_factory()

        result = create_onnxruntime_input(hetero)

        for name, key, attr in EXPECTED_INPUTS:
            value = result[name]
            assert isinstance(value, np.ndarray)
            assert np.array_equal(value, getattr(hetero[key], attr).numpy())


class TestCreateTorchInput:
    def test_returns_the_values_in_input_order(self, hetero_factory):
        hetero = hetero_factory()

        result = create_torch_input(hetero)

        assert isinstance(result, tuple)
        assert len(result) == len(EXPECTED_INPUTS)
        for value, (_name, key, attr) in zip(result, EXPECTED_INPUTS, strict=True):
            assert value is getattr(hetero[key], attr)


class TestInputNames:
    def test_matches_the_expected_wire_names_in_order(self):
        assert input_names() == [name for name, _, _ in EXPECTED_INPUTS]


class TestDynamicAxes:
    def test_matches_the_legacy_dynamic_axes_mapping(self):
        # The dim names are part of the ONNX wire contract (the V# engine
        # matches dynamic dims by name); pinned as literals.
        assert dynamic_axes() == {
            "game_vertex": {0: "num_game_vertices"},
            "state_vertex": {0: "num_state_vertices"},
            "path_condition_vertex": {0: "num_pc_vertices"},
            "gamevertex_to_gamevertex_index": {1: "num_gv_gv_edges"},
            "gamevertex_to_gamevertex_type": {0: "num_gv_gv_edges"},
            "gamevertex_history_statevertex_index": {1: "num_gv_history_sv_edges"},
            "gamevertex_history_statevertex_attrs": {0: "num_gv_history_sv_edges"},
            "gamevertex_in_statevertex": {1: "num_gv_in_sv_edges"},
            "statevertex_parentof_statevertex": {1: "num_sv_parentof_sv_edges"},
            "pathconditionvertex_to_pathconditionvertex": {1: "num_pc_pc_edges"},
            "pathconditionvertex_to_statevertex": {1: "num_pc_sv_edges"},
        }


class TestVerifyOutputs:
    def test_matching_outputs_pass(self):
        out = torch.randn(4, 1)

        verify_outputs(out, out.numpy())

    def test_within_tolerance_passes(self):
        out = torch.randn(4, 1)

        verify_outputs(out, out.numpy() + 1e-5)

    def test_beyond_tolerance_raises_with_the_max_difference(self):
        out = torch.zeros(4, 1)
        perturbed = out.numpy().copy()
        perturbed[2, 0] = 1e-3

        with pytest.raises(
            ValueError,
            match=(
                r"max absolute difference 1\.000e-03 exceeds the tolerance "
                r"1\.0e-04 \(max difference at flattened index 2\)"
            ),
        ):
            verify_outputs(out, perturbed)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            verify_outputs(torch.zeros(4, 1), np.zeros((3, 1)))

    def test_empty_outputs_pass(self):
        verify_outputs(torch.empty(0, 1), np.empty((0, 1)))


class TestDynamicShapes:
    """The dynamo exporter's ``dynamic_shapes`` derivation (#646).

    ``torch.onnx.Dim`` does not exist in the locked torch (2.7.1), so a
    recording stand-in is monkeypatched in; the derivation logic — one
    Dim instance per symbolic dimension name, shared across the axes
    that carry the name — is what the tests pin.
    """

    @staticmethod
    def _install_fake_dim(monkeypatch):
        class FakeDim:
            def __init__(self, name):
                self.name = name

        monkeypatch.setattr(torch.onnx, "Dim", FakeDim, raising=False)

    def test_covers_every_input_in_model_order(self, monkeypatch):
        self._install_fake_dim(monkeypatch)

        shapes = onyx._dynamic_shapes()

        assert list(shapes) == [name for name, _, _ in EXPECTED_INPUTS]

    def test_shares_one_dim_instance_per_name(self, monkeypatch):
        self._install_fake_dim(monkeypatch)

        shapes = onyx._dynamic_shapes()

        gv_gv = shapes["gamevertex_to_gamevertex_index"][1]
        assert gv_gv is shapes["gamevertex_to_gamevertex_type"][0]
        assert gv_gv.name == "num_gv_gv_edges"
        history = shapes["gamevertex_history_statevertex_index"][1]
        assert history is shapes["gamevertex_history_statevertex_attrs"][0]
        assert history.name == "num_gv_history_sv_edges"

    def test_distinct_names_get_distinct_dims(self, monkeypatch):
        self._install_fake_dim(monkeypatch)

        shapes = onyx._dynamic_shapes()

        assert shapes["game_vertex"][0] is not shapes["state_vertex"][0]
        assert shapes["game_vertex"][0].name == "num_game_vertices"
        assert shapes["state_vertex"][0].name == "num_state_vertices"


class TestDynamoExporterVersionGuard:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("2.7.1+cu126", False),
            ("2.12.0", False),
            ("2.13.0", True),
            ("2.14.1", True),
        ],
    )
    def test_parses_torch_versions(self, monkeypatch, version, expected):
        monkeypatch.setattr(torch, "__version__", version)

        assert onyx._torch_uses_dynamo_exporter() is expected


class TestLoadGamestate:
    def test_loads_a_reference_gamestate(self):
        path = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]
        with open(path) as f:
            expected = GameState.from_dict(json.load(f))

        with open(path) as f:
            actual = load_gamestate(f)

        assert actual == expected
