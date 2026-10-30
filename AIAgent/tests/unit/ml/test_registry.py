"""Unit tests for the model registry (``ml.models.registry``).

The registry is the single model-selection path: lookup by name, reverse
lookup by instance/class, and the forward-contract pin that fails at test
time if a registered model's ``forward`` drifts from ``Model``.
"""

import inspect

import pytest
import torch
from ml.models.NorthernPenguin.model import (
    StateModelEncoder as NorthernPenguinEncoder,
)
from ml.models.registry import (
    MODELS,
    get_model_class,
    get_model_name,
    model_names,
)
from ml.protocols import Model

pytestmark = pytest.mark.unit

SMALL_MODEL_KWARGS = {
    "hidden_channels": 8,
    "num_of_state_features": 6,
    "num_hops_1": 1,
    "num_hops_2": 1,
    "normalization": True,
    "num_pc_layers": 1,
}


def test_get_model_class_returns_the_registered_class():
    assert get_model_class("NorthernPenguin") is NorthernPenguinEncoder


def test_get_model_class_unknown_name_raises_listing_the_known_names():
    with pytest.raises(ValueError, match="Unknown model name 'nope'") as excinfo:
        get_model_class("nope")
    assert "NorthernPenguin" in str(excinfo.value)


def test_get_model_name_resolves_an_instance_and_the_class():
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS)

    assert get_model_name(model) == "NorthernPenguin"
    assert get_model_name(NorthernPenguinEncoder) == "NorthernPenguin"


def test_get_model_name_unregistered_class_raises_listing_the_known_names():
    class Unregistered(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="is not registered") as excinfo:
        get_model_name(Unregistered)
    assert "NorthernPenguin" in str(excinfo.value)


def test_model_names_are_the_sorted_registry_keys():
    assert model_names() == sorted(MODELS)
    assert model_names() == ["NorthernPenguin"]


def test_every_registered_model_matches_the_forward_contract():
    contract = list(inspect.signature(Model.forward).parameters)
    for name in model_names():
        model_class = get_model_class(name)
        assert list(inspect.signature(model_class.forward).parameters) == contract, (
            f"{name} does not match the Model forward contract"
        )
