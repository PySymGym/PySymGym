"""Unit tests for the pure ``ml.dataset`` transforms and filtering helpers.

Every graph input is built through ``convert_input_to_tensor`` (via the shared
``gamestate_factory``/``hetero_factory`` fixtures) so the tests cannot drift
from the production tensor schema; ``tmp_dataset`` provides an empty
``TrainingDataset`` for the instance-level filtering helpers.
"""

import pytest
import torch
from ml.dataset import (
    Result,
    convert_input_to_tensor,
    flatten_dict,
    get_hetero_data,
    remove_extra_attrs,
)
from ml.inference import TORCH
from torch_geometric.data import HeteroData

pytestmark = pytest.mark.unit


def test_flatten_dict_concatenates_values_in_order():
    assert flatten_dict({"a": [1, 2], "b": [3]}) == [1, 2, 3]
    assert flatten_dict({}) == []


def test_flatten_dict_returns_a_fresh_list():
    source = {"a": [1]}
    flattened = flatten_dict(source)
    flattened.append(2)
    assert source == {"a": [1]}


def test_remove_extra_attrs_deletes_present_attributes(hetero_factory):
    data = hetero_factory()
    data[*TORCH.gamevertex_to_gamevertex].edge_attr = torch.zeros(1, 2)
    data.use_for_train = True

    remove_extra_attrs(data)

    assert not hasattr(data[*TORCH.statevertex_history_gamevertex], "edge_attr")
    assert not hasattr(data[*TORCH.gamevertex_to_gamevertex], "edge_attr")
    assert not hasattr(data, "use_for_train")


def test_remove_extra_attrs_is_a_noop_without_them(hetero_factory):
    data = hetero_factory()

    remove_extra_attrs(data)

    assert not hasattr(data[*TORCH.gamevertex_to_gamevertex], "edge_attr")
    assert not hasattr(data, "use_for_train")


def test_get_hetero_data_sets_y_true_as_a_column(gamestate_factory):
    data = get_hetero_data(gamestate_factory(), [[0.25, 0.75]])

    assert data["y_true"].shape == (2, 1)
    assert torch.equal(data["y_true"], torch.tensor([[0.25], [0.75]]))
    assert data[TORCH.game_vertex].x.shape[0] == 3


def test_get_hetero_data_keeps_the_converted_graph(gamestate_factory):
    data = get_hetero_data(gamestate_factory(), [[1.0, 0.0]])
    expected, _ = convert_input_to_tensor(gamestate_factory())

    assert torch.equal(data[TORCH.game_vertex].x, expected[TORCH.game_vertex].x)
    assert torch.equal(data[TORCH.state_vertex].x, expected[TORCH.state_vertex].x)


def _step(y_true: torch.Tensor) -> HeteroData:
    step = HeteroData()
    step["y_true"] = y_true
    return step


def test_filter_map_steps_keeps_multi_state_rows_and_one_hot_encodes(tmp_dataset):
    single_state = _step(torch.tensor([[1.0]]))
    with_nan = _step(torch.tensor([[float("nan")], [1.0]]))
    multi_state = _step(torch.tensor([[0.2], [0.8]]))

    kept = tmp_dataset.filter_map_steps([single_state, with_nan, multi_state])

    assert len(kept) == 1
    assert torch.equal(kept[0]["y_true"], torch.tensor([[0.0], [1.0]]))


def test_is_update_map_required_for_unseen_map(tmp_dataset):
    assert tmp_dataset.is_update_map_required("m", Result(50, 0, 0, 0)) is True


def test_is_update_map_required_for_reproduced_full_coverage(tmp_dataset):
    tmp_dataset.maps_results = {"m": Result(100, 0, 0, 0)}

    assert tmp_dataset.is_update_map_required("m", Result(100, 0, 0, 0)) is True


def test_is_update_map_required_for_equal_result_below_full_coverage(tmp_dataset):
    tmp_dataset.maps_results = {"m": Result(50, 0, 0, 0)}

    assert tmp_dataset.is_update_map_required("m", Result(50, 0, 0, 0)) is False


def test_is_update_map_required_for_better_result(tmp_dataset):
    tmp_dataset.maps_results = {"m": Result(50, 0, 0, 0)}

    assert tmp_dataset.is_update_map_required("m", Result(60, 0, 0, 0)) is True


def test_is_update_map_required_for_worse_result(tmp_dataset):
    tmp_dataset.maps_results = {"m": Result(60, 0, 0, 0)}

    assert tmp_dataset.is_update_map_required("m", Result(50, 0, 0, 0)) is False
