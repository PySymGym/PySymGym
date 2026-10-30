"""Unit tests for the pure ``ml.dataset`` transforms and filtering helpers.

Every graph input is built through ``convert_input_to_tensor`` (via the shared
``gamestate_factory``/``hetero_factory`` fixtures) so the tests cannot drift
from the production tensor schema; ``tmp_dataset`` provides an empty
``TrainingDataset`` for the instance-level filtering helpers.
"""

import pytest
import torch
from common.dataset import Result
from ml.dataset import (
    convert_input_to_tensor,
    flatten_dict,
    get_hetero_data,
    remove_extra_attrs,
)
from ml.inference import TORCH
from ml.schema import NUM_STATE_FEATURES
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


def test_convert_input_to_tensor_rejects_duplicate_pc_vertex_id(gamestate_factory):
    gs = gamestate_factory()
    gs.PathConditionVertices[1].Id = 0

    with pytest.raises(
        RuntimeError, match="There are two path condition vertices with the same Id."
    ):
        convert_input_to_tensor(gs)


def test_convert_input_to_tensor_deduplicates_state_nodes_but_not_child_edges(
    gamestate_factory,
):
    gs = gamestate_factory()
    gs.States.append(gs.States[0])  # duplicate Id 0

    dup_graph, dup_map = convert_input_to_tensor(gs)
    ref_graph, ref_map = convert_input_to_tensor(gamestate_factory())

    assert dup_map == ref_map
    assert torch.equal(dup_graph[TORCH.game_vertex].x, ref_graph[TORCH.game_vertex].x)
    assert torch.equal(dup_graph[TORCH.state_vertex].x, ref_graph[TORCH.state_vertex].x)
    assert torch.equal(
        dup_graph[TORCH.path_condition_vertex].x,
        ref_graph[TORCH.path_condition_vertex].x,
    )
    # The duplicated state contributes its child edges a second time.
    assert torch.equal(
        dup_graph[*TORCH.statevertex_parentof_statevertex].edge_index,
        torch.tensor([[0, 0], [1, 1]]),
    )
    for edge_type in (
        TORCH.gamevertex_to_gamevertex,
        TORCH.statevertex_in_gamevertex,
        TORCH.gamevertex_in_statevertex,
        TORCH.statevertex_history_gamevertex,
        TORCH.gamevertex_history_statevertex,
        TORCH.pathcondvertex_to_pathcondvertex,
        TORCH.pathcondvertex_to_statevertex,
        TORCH.statevertex_to_pathcondvertex,
    ):
        assert torch.equal(
            dup_graph[edge_type].edge_index, ref_graph[edge_type].edge_index
        ), edge_type


def test_convert_input_to_tensor_returns_none_on_missing_child_state(
    gamestate_factory, capsys
):
    gs = gamestate_factory()
    gs.States[0].Children = [99]  # no state with Id 99

    graph, state_map = convert_input_to_tensor(gs)

    assert graph is None
    assert state_map is None
    assert "[ERROR]: KeyError" in capsys.readouterr().out


def _step(y_true: torch.Tensor, state_rows: int = 2, game_rows: int = 3) -> HeteroData:
    step = HeteroData()
    step[TORCH.state_vertex].x = torch.zeros(state_rows, NUM_STATE_FEATURES)
    step[TORCH.game_vertex].x = torch.zeros(game_rows, 7)
    step["y_true"] = y_true
    return step


def test_filter_map_steps_keeps_multi_state_rows_and_one_hot_encodes(tmp_dataset):
    single_state = _step(torch.tensor([[1.0]]))
    with_nan = _step(torch.tensor([[float("nan")], [1.0]]))
    multi_state = _step(torch.tensor([[0.2], [0.8]]))

    kept = tmp_dataset.filter_map_steps([single_state, with_nan, multi_state])

    assert len(kept) == 1
    assert torch.equal(kept[0]["y_true"], torch.tensor([[0.0], [1.0]]))


def test_remove_similar_steps_keeps_steps_with_differing_state_counts(tmp_dataset):
    first = _step(torch.tensor([[1.0], [0.0]]), state_rows=2)
    second = _step(torch.tensor([[1.0], [0.0], [0.0]]), state_rows=3)

    kept = tmp_dataset.remove_similar_steps([first, second])

    assert kept == [first, second]


def test_remove_similar_steps_keeps_steps_with_differing_game_vertex_counts(
    tmp_dataset,
):
    first = _step(torch.tensor([[1.0], [0.0]]), game_rows=3)
    second = _step(torch.tensor([[1.0], [0.0]]), game_rows=4)

    kept = tmp_dataset.remove_similar_steps([first, second])

    assert kept == [first, second]


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


# ---------------------------------------------------------------------------
# Result ordering (#594)
#
# Convention: ``a < b`` means *b is better* than a (the call-site
# convention of ``is_update_map_required``/``update_map``). Better means:
# higher coverage, then fewer tests, then fewer steps, then fewer errors.
# ---------------------------------------------------------------------------


def test_result_ordering_more_errors_ranks_worse():
    # The #594 regression: at equal coverage/tests/steps, the errorful run
    # must rank worse than the error-free one (it used to rank better).
    assert Result(100, 5, 10, 3) < Result(100, 5, 10, 0)
    assert not (Result(100, 5, 10, 0) < Result(100, 5, 10, 3))


def test_result_ordering_fewer_tests_rank_better():
    assert Result(100, 2, 5, 0) < Result(100, 1, 5, 0)
    assert not (Result(100, 1, 5, 0) < Result(100, 2, 5, 0))


def test_result_ordering_fewer_steps_rank_better():
    assert Result(100, 1, 6, 0) < Result(100, 1, 5, 0)
    assert not (Result(100, 1, 5, 0) < Result(100, 1, 6, 0))


def test_result_ordering_coverage_dominates():
    assert Result(99, 0, 0, 0) < Result(100, 9, 9, 9)


def test_result_ordering_mixed_comparison_raises_type_error():
    with pytest.raises(TypeError):
        _ = Result(100, 1, 5, 0) < 42


def test_result_to_from_tuple_round_trips_the_disk_format():
    assert Result(100, 1, 5, 0).to_tuple() == (100, -1, -5, 0)
    assert Result.from_tuple((100, -1, -5, 0)) == Result(100, 1, 5, 0)


def test_is_update_map_required_rejects_more_errors_at_equal_rest(tmp_dataset):
    # The defect scenario: a run with errors must not replace an error-free
    # run at equal coverage/tests/steps.
    tmp_dataset.maps_results = {"m": Result(100, 5, 10, 0)}

    assert tmp_dataset.is_update_map_required("m", Result(100, 5, 10, 3)) is False


def test_is_update_map_required_accepts_fewer_errors_at_equal_rest(tmp_dataset):
    tmp_dataset.maps_results = {"m": Result(100, 5, 10, 3)}

    assert tmp_dataset.is_update_map_required("m", Result(100, 5, 10, 0)) is True


def test_is_update_map_required_accepts_fewer_tests_or_steps(tmp_dataset):
    tmp_dataset.maps_results = {"m": Result(100, 5, 10, 0)}

    assert tmp_dataset.is_update_map_required("m", Result(100, 4, 10, 0)) is True
    assert tmp_dataset.is_update_map_required("m", Result(100, 5, 9, 0)) is True
