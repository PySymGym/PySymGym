"""Unit tests for ``ml.predict`` and ``ml.training.wrapper``.

``predict_state_with_dict`` is driven with a fixed-output fake model so the
remapping and max-row-sum selection are pinned exactly, plus one real-model
run through the production ``hetero_factory`` graph. ``TrainingModelWrapper``
is pinned through a monkeypatched ``predict_state_with_dict`` recorder (the
delegation contract) and one real-model ``predict`` run.
"""

import inspect

import pytest
import torch
from ml.dataset import convert_input_to_tensor
from ml.models.NorthernPenguin.model import StateModelEncoder as NorthernPenguinEncoder
from ml.predict import predict_state_with_dict
from ml.protocols import Predictor
from ml.training.wrapper import TrainingModelWrapper

pytestmark = pytest.mark.unit

SMALL_MODEL_KWARGS = {
    "hidden_channels": 8,
    "num_of_state_features": 6,
    "num_hops_1": 1,
    "num_hops_2": 1,
    "normalization": True,
    "num_pc_layers": 1,
}


class FixedOutputModel(torch.nn.Module):
    """A model whose output is a fixed tensor, independent of the input."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, **kwargs) -> torch.Tensor:
        return self.output


def test_predict_state_with_dict_selects_the_max_sum_row_and_returns_the_output(
    hetero_factory,
) -> None:
    model = FixedOutputModel(torch.tensor([[0.0], [2.0]]))
    data = hetero_factory()

    next_step_id, out = predict_state_with_dict(model, data, {0: 0, 1: 1})

    assert next_step_id == 1
    assert torch.equal(out, torch.tensor([[0.0], [2.0]]))
    assert out.requires_grad is False


def test_predict_state_with_dict_remaps_indices_through_the_state_map(
    hetero_factory,
) -> None:
    model = FixedOutputModel(torch.tensor([[0.0], [2.0]]))
    data = hetero_factory()

    next_step_id, _ = predict_state_with_dict(model, data, {5: 0, 9: 1})

    assert next_step_id == 9


def test_predict_state_with_dict_runs_a_real_model_without_gradients(
    hetero_factory,
) -> None:
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS)
    data = hetero_factory()
    state_map = {0: 0, 1: 1}

    next_step_id, out = predict_state_with_dict(model, data, state_map)

    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()
    assert out.requires_grad is False
    expected = max(
        state_map, key=lambda state_id: out[state_map[state_id]].sum().item()
    )
    assert next_step_id == expected


def test_wrapper_name_and_model_accessor() -> None:
    model = torch.nn.Linear(2, 1)
    wrapper = TrainingModelWrapper(model)

    assert wrapper.name() == "Training model"
    assert wrapper.model() is model
    # Until make_copy is called, model_copy is the model itself.
    assert wrapper.model_copy is model


def test_wrapper_predict_signature_matches_the_protocol() -> None:
    # Python does not enforce ABC signatures: a drift on either side would
    # only surface at a call site. Pin the parameter lists equal.
    assert list(inspect.signature(TrainingModelWrapper.predict).parameters) == list(
        inspect.signature(Predictor.predict).parameters
    )


def test_wrapper_make_copy_deepcopies_the_model_and_renames_it() -> None:
    model = torch.nn.Linear(2, 1)
    wrapper = TrainingModelWrapper(model)

    wrapper.make_copy("two")

    assert wrapper.model_copy is not model
    assert torch.equal(wrapper.model().weight, wrapper.model_copy.weight)
    assert wrapper._name == "two"
    with torch.no_grad():
        wrapper.model_copy.weight.add_(1.0)
    assert not torch.equal(wrapper.model().weight, wrapper.model_copy.weight)


def test_wrapper_predict_delegates_to_predict_state_with_dict(
    monkeypatch: pytest.MonkeyPatch, gamestate_factory
) -> None:
    model = torch.nn.Linear(2, 1)
    wrapper = TrainingModelWrapper(model)
    sentinel = object()
    recorded: dict = {}

    def fake_predict(model_arg, data_arg, state_map_arg):
        recorded["model"] = model_arg
        recorded["data"] = data_arg
        recorded["state_map"] = state_map_arg
        return 7, sentinel

    monkeypatch.setattr("ml.training.wrapper.predict_state_with_dict", fake_predict)

    result = wrapper.predict(gamestate_factory())

    assert result == (7, sentinel)
    assert recorded["model"] is model
    assert recorded["state_map"] == {0: 0, 1: 1}
    # The graph is the production conversion of the gamestate, unchanged.
    expected_graph, _ = convert_input_to_tensor(gamestate_factory())
    assert set(recorded["data"].node_types) == set(expected_graph.node_types)
    for node_type in expected_graph.node_types:
        assert torch.equal(recorded["data"][node_type].x, expected_graph[node_type].x)
    for edge_type in expected_graph.edge_types:
        assert torch.equal(
            recorded["data"][edge_type].edge_index,
            expected_graph[edge_type].edge_index,
        )


def test_wrapper_predict_runs_the_real_model(gamestate_factory) -> None:
    model = NorthernPenguinEncoder(**SMALL_MODEL_KWARGS)
    wrapper = TrainingModelWrapper(model)

    next_step_id, out = wrapper.predict(gamestate_factory())

    assert next_step_id in (0, 1)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()
