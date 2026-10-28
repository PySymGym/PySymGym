"""Unit tests for the pure helpers of the ONNX export entrypoint ``onyx.py``.

These pin the current (pre-#578) behaviour of the tensor-input builders, the
output shortener, the gamestate loader and the model-FQN resolver. The ONNX
input names are pinned as literal strings: they are the V# wire contract and
the drift net for #597's single-source input-names change.
"""

import json

import numpy as np
import pytest
import torch
from common.game import GameState
from ml.inference import TORCH
from ml.models.NorthernPenguin.model import (
    StateModelEncoder as RealStateModelEncoder,
)
from onyx import (
    create_model_input,
    create_onnxruntime_input,
    create_torch_input,
    load_gamestate,
    resolve_import_model,
    shorten_output,
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


class TestShortenOutput:
    def test_flattens_and_rounds_to_three_decimals(self):
        out = torch.tensor([[1.23456], [2.0], [-0.5]])

        assert shorten_output(out) == [1.235, 2.0, -0.5]

    def test_truncates_longer_than_ten_elements_with_ellipsis(self):
        # The truncation is part of the #597 weak verification (a divergence
        # beyond element 10 passes the current assert); the helper's contract
        # is pinned as current behaviour and #597 owns updating this pin if
        # the helper changes.
        out = torch.arange(12, dtype=torch.float)

        assert shorten_output(out) == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            "...",
        ]

    def test_exactly_ten_elements_are_not_truncated(self):
        out = torch.arange(10, dtype=torch.float)

        assert shorten_output(out) == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ]

    def test_custom_round_to(self):
        out = torch.tensor([1.23456])

        assert shorten_output(out, round_to=1) == [1.2]


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


class TestLoadGamestate:
    def test_loads_a_reference_gamestate(self):
        path = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]
        with open(path) as f:
            expected = GameState.from_dict(json.load(f))

        with open(path) as f:
            actual = load_gamestate(f)

        assert actual == expected


class TestResolveImportModel:
    def test_resolves_a_valid_fqn(self):
        model_def = resolve_import_model(
            "ml.models.NorthernPenguin.model.StateModelEncoder"
        )

        assert model_def is RealStateModelEncoder

    def test_missing_module_raises_module_not_found(self):
        with pytest.raises(ModuleNotFoundError):
            resolve_import_model("no.such.module.StateModelEncoder")

    def test_missing_class_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            resolve_import_model("ml.models.NorthernPenguin.model.NoSuchClass")
