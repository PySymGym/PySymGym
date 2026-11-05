"""Unit tests for the ``ml.protocols`` abstract interfaces.

``Named`` lives in ``common.protocols`` (the shared layer must not import
``ml``); ``ml.protocols`` re-exports it for ``Predictor``.
"""

import inspect

import pytest
from common.protocols import Named
from ml.protocols import Model, Predictor, Named as MlNamed

pytestmark = pytest.mark.unit


def test_named_is_reexported_by_ml_protocols():
    # One class, two import paths: the ml-side name is the common-side one.
    assert MlNamed is Named


def test_named_and_predictor_are_abstract():
    assert inspect.isabstract(Named)
    assert inspect.isabstract(Predictor)

    with pytest.raises(TypeError):
        Named()
    with pytest.raises(TypeError):
        Predictor()


def test_predictor_requires_name_predict_and_model():
    assert Predictor.__abstractmethods__ == frozenset({"name", "predict", "model"})


def test_abstract_bodies_raise_not_implemented_error():
    # The abstract bodies are unreachable through instances (instantiation
    # is impossible by design); invoking them unbound on a sentinel is the
    # only executable path to those lines.
    sentinel = object()
    with pytest.raises(NotImplementedError):
        Named.name(sentinel)
    with pytest.raises(NotImplementedError):
        Predictor.predict(sentinel, input=None)
    with pytest.raises(NotImplementedError):
        Predictor.model(sentinel)


def test_concrete_predictor_satisfies_the_interface():
    class DummyPredictor(Predictor):
        def name(self) -> str:
            return "dummy"

        def predict(self, input):
            return input

        def model(self):
            return None

    predictor = DummyPredictor()
    assert not inspect.isabstract(DummyPredictor)
    assert predictor.name() == "dummy"
    assert predictor.predict(input=None) is None
    assert predictor.model() is None


def test_model_protocol_declares_the_infer_contract():
    # The 11-argument forward signature ``ml.inference.infer`` passes, in
    # order: the contract every registered model must satisfy.
    assert list(inspect.signature(Model.forward).parameters) == [
        "self",
        "game_x",
        "state_x",
        "pc_x",
        "edge_index_v_v",
        "edge_type_v_v",
        "edge_index_history_v_s",
        "edge_attr_history_v_s",
        "edge_index_in_v_s",
        "edge_index_s_s",
        "edge_index_pc_pc",
        "edge_index_pc_s",
    ]
