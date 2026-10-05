"""Unit tests for the ``ml.protocols`` abstract interfaces."""

import inspect

import pytest
from ml.protocols import Named, Predictor

pytestmark = pytest.mark.unit


def test_named_and_predictor_are_abstract():
    assert inspect.isabstract(Named)
    assert inspect.isabstract(Predictor)

    with pytest.raises(TypeError):
        Named()
    with pytest.raises(TypeError):
        Predictor()


def test_predictor_requires_name_predict_and_model():
    assert Predictor.__abstractmethods__ == frozenset({"name", "predict", "model"})


def test_concrete_predictor_satisfies_the_interface():
    class DummyPredictor(Predictor):
        def name(self) -> str:
            return "dummy"

        def predict(self, input, map_name):
            return map_name

        def model(self):
            return None

    predictor = DummyPredictor()
    assert not inspect.isabstract(DummyPredictor)
    assert predictor.name() == "dummy"
    assert predictor.predict(input=None, map_name="map") == "map"
    assert predictor.model() is None
