"""Unit tests for ``TrainingDataset`` processing and maintenance.

``process()`` is exercised through the ``serial_pool`` fixture: forked
``mp.Pool`` workers are invisible to pytest-cov, and the pool is an
orchestration seam, not the behaviour under test. The raw dataset layout
mirrors what the game servers serialize: per map a ``result`` file
(space-separated ``coverage tests steps errors``, all non-negative) and, per
step, ``<id>_gameState`` (JSON ``GameState``) plus ``<id>_movedState``
(decimal state Id).
"""

import json
import logging
from pathlib import Path

import pytest
import torch
from common.dataset import Result
from ml.dataset import TrainingDataset, convert_input_to_tensor
from ml.inference import TORCH

pytestmark = pytest.mark.unit


def _populate_raw_map(
    raw: Path,
    map_name: str,
    gamestate_factory,
    steps: dict[str, int],
    result: str = "100 1 5 0",
) -> Path:
    map_dir = raw / map_name
    map_dir.mkdir(parents=True)
    (map_dir / "result").write_text(result)
    for step_id, moved_state in steps.items():
        (map_dir / f"{step_id}_gameState").write_text(
            json.dumps(gamestate_factory().to_dict())
        )
        (map_dir / f"{step_id}_movedState").write_text(str(moved_state))
    return map_dir


def _load_step(path: Path):
    return torch.load(path, weights_only=False)


def test_construction_auto_processes_the_raw_dir(
    tmp_path, gamestate_factory, serial_pool
):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _populate_raw_map(raw, "mapA", gamestate_factory, {"7": 0})
    processed.mkdir()

    ds = TrainingDataset(
        raw_dir=raw, processed_dir=processed, train_percentage=1.0, n_jobs=1
    )

    map_dir = processed / "mapA"
    assert sorted(p.name for p in map_dir.iterdir()) == ["0.pt", "result"]
    assert (map_dir / "result").read_text() == "(100, -1, -5, 0)"
    assert ds.maps_results == {"mapA": Result(100, 1, 5, 0)}
    assert ds.processed_paths == [str(map_dir / "0.pt")]
    assert torch.equal(
        _load_step(map_dir / "0.pt")["y_true"], torch.tensor([[1.0], [0.0]])
    )


def test_process_orders_steps_by_sorted_raw_id(
    tmp_path, gamestate_factory, serial_pool
):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    ds = TrainingDataset(
        raw_dir=raw, processed_dir=processed, train_percentage=1.0, n_jobs=1
    )
    # The auto-processing run on the empty raw dir is a no-op.
    assert ds.processed_paths == []

    _populate_raw_map(raw, "mapA", gamestate_factory, {"7": 0, "3": 1})
    ds.process()

    map_dir = processed / "mapA"
    # Raw ids are sorted: "3" -> 0.pt (moved state 1), "7" -> 1.pt (moved state 0).
    assert torch.equal(
        _load_step(map_dir / "0.pt")["y_true"], torch.tensor([[0.0], [1.0]])
    )
    assert torch.equal(
        _load_step(map_dir / "1.pt")["y_true"], torch.tensor([[1.0], [0.0]])
    )
    assert (map_dir / "result").read_text() == "(100, -1, -5, 0)"
    assert ds.maps_results == {"mapA": Result(100, 1, 5, 0)}
    assert ds.processed_paths == [str(map_dir / "0.pt"), str(map_dir / "1.pt")]


def test_process_and_save_step_saves_the_converted_graph(
    tmp_path, gamestate_factory, serial_pool
):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    ds = TrainingDataset(
        raw_dir=raw, processed_dir=processed, train_percentage=1.0, n_jobs=1
    )
    gs = gamestate_factory()
    map_dir = _populate_raw_map(raw, "mapA", gamestate_factory, {"7": 1})
    out_dir = processed / "out"
    out_dir.mkdir()

    assert ds.process_and_save_step(map_dir, out_dir, ("7", 5)) is True

    step = _load_step(out_dir / "5.pt")
    ref_graph, _ = convert_input_to_tensor(gs)
    assert torch.equal(step[TORCH.game_vertex].x, ref_graph[TORCH.game_vertex].x)
    assert torch.equal(step[TORCH.state_vertex].x, ref_graph[TORCH.state_vertex].x)
    assert torch.equal(
        step[TORCH.path_condition_vertex].x,
        ref_graph[TORCH.path_condition_vertex].x,
    )
    assert torch.equal(step["y_true"], torch.tensor([[0.0], [1.0]]))


def test_double_process_is_idempotent(tmp_path, gamestate_factory, serial_pool):
    # The FileExistsError pin is retired: the processed result file is now
    # overwritten, and the .pt steps are rewritten in place, so re-running
    # process() over an already-processed map converges to the same
    # on-disk state.
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _populate_raw_map(raw, "mapA", gamestate_factory, {"7": 0})
    processed.mkdir()
    ds = TrainingDataset(
        raw_dir=raw, processed_dir=processed, train_percentage=1.0, n_jobs=1
    )
    # Construction already auto-processed mapA.
    map_dir = processed / "mapA"
    result_before = (map_dir / "result").read_text()
    y_true_before = _load_step(map_dir / "0.pt")["y_true"]

    ds.process()

    assert sorted(p.name for p in map_dir.iterdir()) == ["0.pt", "result"]
    assert (map_dir / "result").read_text() == result_before
    assert torch.equal(_load_step(map_dir / "0.pt")["y_true"], y_true_before)


# ---------------------------------------------------------------------------
# update_map / _save_steps branch matrix
#
# Steps come from ``step_factory`` (the deterministic ``hetero_factory``
# graph plus a ``y_true`` distribution). ``filter_map_steps`` binarizes
# ``y_true`` at its argmax, so the tensors below are already one-hot;
# ``remove_similar_steps`` drops a step only when its distribution and
# game-vertex count match the previous kept step (at
# ``similar_steps_save_prob=0``), so the sequences below survive intact.
# The exact ``logging.info`` messages are pinned via caplog, including the
# trailing space of the merge/reproduce messages.
# ---------------------------------------------------------------------------


def test_update_map_saves_a_new_map(tmp_dataset, step_factory, serial_pool, caplog):
    map_name = "mapA"
    result = Result(100, 1, 5, 0)
    steps = [
        step_factory(torch.tensor([[1.0], [0.0]])),
        step_factory(torch.tensor([[0.0], [1.0]])),
    ]

    with caplog.at_level(logging.INFO):
        tmp_dataset.update_map(map_name, result, steps)

    map_dir = tmp_dataset.processed_dir / map_name
    assert sorted(p.name for p in map_dir.iterdir()) == ["0.pt", "1.pt", "result"]
    assert (map_dir / "result").read_text() == "(100, -1, -5, 0)"
    assert tmp_dataset.maps_results == {map_name: result}
    assert torch.equal(
        _load_step(map_dir / "0.pt")["y_true"], torch.tensor([[1.0], [0.0]])
    )
    assert torch.equal(
        _load_step(map_dir / "1.pt")["y_true"], torch.tensor([[0.0], [1.0]])
    )
    assert caplog.messages == [
        f"New map with name {map_name} was saved with result {tuple(result)}"
    ]


def test_update_map_below_threshold_saves_result_only(
    tmp_dataset, step_factory, serial_pool, caplog
):
    map_name = "mapA"
    result = Result(50, 0, 0, 0)

    with caplog.at_level(logging.INFO):
        tmp_dataset.update_map(
            map_name, result, [step_factory(torch.tensor([[1.0], [0.0]]))]
        )

    map_dir = tmp_dataset.processed_dir / map_name
    assert [p.name for p in map_dir.iterdir()] == ["result"]
    assert (map_dir / "result").read_text() == "(50, 0, 0, 0)"
    assert tmp_dataset.maps_results == {map_name: result}
    assert caplog.messages == [
        f"New map with name {map_name} was saved with result {tuple(result)}"
    ]


def test_update_map_merges_matching_steps(
    tmp_dataset, step_factory, serial_pool, caplog
):
    map_name = "mapA"
    result = Result(100, 1, 5, 0)
    tmp_dataset.update_map(
        map_name, result, [step_factory(torch.tensor([[1.0], [0.0]]))]
    )

    # Same graph as the stored step, opposite distribution -> merged into
    # the stored step. Mutated graph, distribution of the stored step ->
    # survives remove_similar_steps (differs from the previous kept step)
    # and is appended as a unique step.
    matched = step_factory(torch.tensor([[0.0], [1.0]]))
    unique = step_factory(torch.tensor([[1.0], [0.0]]))
    unique[TORCH.game_vertex].x[0, 0] += 1.0

    with caplog.at_level(logging.INFO):
        tmp_dataset.update_map(map_name, result, [matched, unique])

    map_dir = tmp_dataset.processed_dir / map_name
    assert sorted(p.name for p in map_dir.iterdir()) == ["0.pt", "1.pt", "result"]
    # Unique new steps first, then all old steps: 0.pt is the unique step
    # (mutated graph, its own distribution), 1.pt is the old step with the
    # merged distribution [1, 0] + [0, 1] -> [0.5, 0.5].
    expected_unique_x = matched[TORCH.game_vertex].x.clone()
    expected_unique_x[0, 0] += 1.0
    assert torch.equal(
        _load_step(map_dir / "0.pt")[TORCH.game_vertex].x, expected_unique_x
    )
    assert torch.equal(
        _load_step(map_dir / "0.pt")["y_true"], torch.tensor([[1.0], [0.0]])
    )
    assert torch.equal(
        _load_step(map_dir / "1.pt")["y_true"], torch.tensor([[0.5], [0.5]])
    )
    assert caplog.messages == [
        f"Steps on map {map_name} were merged with current steps with result "
        f"{tuple(result)}. 2 + 1 -> 2. "
    ]


def test_update_map_reproduced_branch(tmp_dataset, step_factory, serial_pool, caplog):
    map_name = "mapA"
    result = Result(100, 1, 5, 0)
    tmp_dataset.update_map(
        map_name, result, [step_factory(torch.tensor([[1.0], [0.0]]))]
    )

    # Graph-identical step with the same distribution: merged into the
    # stored step ([1, 0] + [1, 0] -> [2, 0], binarized to [1, 0], divided
    # by its mass 1 -> [1, 0], i.e. unchanged), nothing appended, so the
    # step count does not grow and the "reproduced" branch is taken.
    with caplog.at_level(logging.INFO):
        tmp_dataset.update_map(
            map_name, result, [step_factory(torch.tensor([[1.0], [0.0]]))]
        )

    map_dir = tmp_dataset.processed_dir / map_name
    assert sorted(p.name for p in map_dir.iterdir()) == ["0.pt", "result"]
    assert torch.equal(
        _load_step(map_dir / "0.pt")["y_true"], torch.tensor([[1.0], [0.0]])
    )
    assert caplog.messages == [
        f"Existing results reproduced on map {map_name} with result "
        f"{tuple(result)}. 1 + 1 -> 1. "
    ]


def test_update_map_replaces_a_worse_result(
    tmp_dataset, step_factory, serial_pool, caplog
):
    map_name = "mapA"
    old_result = Result(50, 0, 0, 0)
    new_result = Result(100, 1, 5, 0)
    tmp_dataset.update_map(
        map_name, old_result, [step_factory(torch.tensor([[1.0], [0.0]]))]
    )
    # The below-threshold map holds only the result file.
    assert [p.name for p in (tmp_dataset.processed_dir / map_name).iterdir()] == [
        "result"
    ]

    with caplog.at_level(logging.INFO):
        tmp_dataset.update_map(
            map_name, new_result, [step_factory(torch.tensor([[0.0], [1.0]]))]
        )

    map_dir = tmp_dataset.processed_dir / map_name
    assert sorted(p.name for p in map_dir.iterdir()) == ["0.pt", "result"]
    assert (map_dir / "result").read_text() == "(100, -1, -5, 0)"
    assert tmp_dataset.maps_results == {map_name: new_result}
    assert torch.equal(
        _load_step(map_dir / "0.pt")["y_true"], torch.tensor([[0.0], [1.0]])
    )
    assert caplog.messages == [
        f"The model with result = {tuple(old_result)} was replaced with the model with "
        f"result = {tuple(new_result)} on the map {map_name}"
    ]


def test_update_map_no_op_on_equal_sub_threshold_result(
    tmp_dataset, step_factory, serial_pool, caplog
):
    # Pinned current behaviour (finding on #632): an equal result with
    # coverage below 100 takes neither the merge nor the replace branch, so
    # nothing is saved and no log line is emitted.
    map_name = "mapA"
    result = Result(50, 0, 0, 0)
    tmp_dataset.update_map(
        map_name, result, [step_factory(torch.tensor([[1.0], [0.0]]))]
    )

    with caplog.at_level(logging.INFO):
        tmp_dataset.update_map(
            map_name, result, [step_factory(torch.tensor([[0.0], [1.0]]))]
        )

    map_dir = tmp_dataset.processed_dir / map_name
    assert [p.name for p in map_dir.iterdir()] == ["result"]
    assert (map_dir / "result").read_text() == "(50, 0, 0, 0)"
    assert tmp_dataset.maps_results == {map_name: result}
    assert caplog.messages == []


def test_update_map_no_op_on_worse_result(
    tmp_dataset, step_factory, serial_pool, caplog
):
    # Pinned current behaviour (finding on #632): a stored result that is
    # not worse than the incoming one takes no branch, so the stored steps
    # and result are untouched.
    map_name = "mapA"
    result = Result(100, 1, 5, 0)
    tmp_dataset.update_map(
        map_name, result, [step_factory(torch.tensor([[1.0], [0.0]]))]
    )

    with caplog.at_level(logging.INFO):
        tmp_dataset.update_map(
            map_name, Result(50, 0, 0, 0), [step_factory(torch.tensor([[0.0], [1.0]]))]
        )

    map_dir = tmp_dataset.processed_dir / map_name
    assert sorted(p.name for p in map_dir.iterdir()) == ["0.pt", "result"]
    assert (map_dir / "result").read_text() == "(100, -1, -5, 0)"
    assert tmp_dataset.maps_results == {map_name: result}
    assert torch.equal(
        _load_step(map_dir / "0.pt")["y_true"], torch.tensor([[1.0], [0.0]])
    )
    assert caplog.messages == []


def test_update_map_updates_the_load_to_cpu_cache(tmp_path, step_factory, serial_pool):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    ds = TrainingDataset(
        raw_dir=raw,
        processed_dir=processed,
        train_percentage=1.0,
        load_to_cpu=True,
        n_jobs=1,
    )
    steps = [step_factory(torch.tensor([[1.0], [0.0]]))]

    ds.update_map("mapA", Result(100, 1, 5, 0), steps)

    assert list(ds._loaded_to_cpu) == ["mapA"]
    assert len(ds._loaded_to_cpu["mapA"]) == 1
    assert torch.equal(
        ds._loaded_to_cpu["mapA"][0]["y_true"], torch.tensor([[1.0], [0.0]])
    )


# ---------------------------------------------------------------------------
# _get_map_steps / _merge_steps
#
# ``_merge_steps`` reads the old steps from disk through
# ``_get_map_steps(map_name, device="cpu")``, so the old steps are placed on
# disk via ``update_map`` first. Unlike ``update_map``, ``_merge_steps``
# uses the passed ``y_true`` tensors as-is (no binarization).
# ---------------------------------------------------------------------------


def test_get_map_steps_loads_sorted_pt_files_on_cpu(
    tmp_dataset, step_factory, serial_pool
):
    tmp_dataset.update_map(
        "mapA",
        Result(100, 1, 5, 0),
        [
            step_factory(torch.tensor([[1.0], [0.0]])),
            step_factory(torch.tensor([[0.0], [1.0]])),
        ],
    )

    steps = tmp_dataset._get_map_steps("mapA")

    # The "result" file in the map dir is excluded; the .pt files come back
    # in sorted filename order on the default device (CPU under the
    # cpu_device fixture).
    assert len(steps) == 2
    assert torch.equal(steps[0]["y_true"], torch.tensor([[1.0], [0.0]]))
    assert torch.equal(steps[1]["y_true"], torch.tensor([[0.0], [1.0]]))
    assert steps[0][TORCH.game_vertex].x.device.type == "cpu"


def test_merge_steps_combines_unique_new_and_all_old(
    tmp_dataset, step_factory, serial_pool
):
    tmp_dataset.update_map(
        "mapA", Result(100, 1, 5, 0), [step_factory(torch.tensor([[1.0], [0.0]]))]
    )

    # Same graph as the stored step -> merged into it, not appended.
    # Mutated graph -> appended as a unique step.
    matched = step_factory(torch.tensor([[0.0], [1.0]]))
    unique = step_factory(torch.tensor([[1.0], [0.0]]))
    unique[TORCH.game_vertex].x[0, 0] += 1.0

    merged = tmp_dataset._merge_steps([matched, unique], "mapA")

    # Unique new steps first (input order), then all old steps.
    assert len(merged) == 2
    assert torch.equal(merged[0]["y_true"], torch.tensor([[1.0], [0.0]]))
    assert torch.equal(merged[1]["y_true"], torch.tensor([[0.5], [0.5]]))


def test_merge_steps_appends_a_new_vertex_group_without_comparison(
    tmp_dataset, step_factory, serial_pool
):
    tmp_dataset.update_map(
        "mapA", Result(100, 1, 5, 0), [step_factory(torch.tensor([[1.0], [0.0]]))]
    )

    # Dropping a game-vertex row moves the step into a vertex-count group
    # (state + game rows) absent from the old steps: appended without any
    # graph comparison.
    other_group = step_factory(torch.tensor([[1.0], [0.0]]))
    other_group[TORCH.game_vertex].x = other_group[TORCH.game_vertex].x[:2]

    merged = tmp_dataset._merge_steps([other_group], "mapA")

    assert len(merged) == 2
    assert merged[0][TORCH.game_vertex].x.shape[0] == 2
    assert torch.equal(merged[0]["y_true"], torch.tensor([[1.0], [0.0]]))
    assert torch.equal(merged[1]["y_true"], torch.tensor([[1.0], [0.0]]))


def test_merge_steps_disjoint_distributions(tmp_dataset, step_factory, serial_pool):
    tmp_dataset.update_map(
        "mapA",
        Result(100, 1, 5, 0),
        [step_factory(torch.tensor([[1.0], [0.0], [0.0]]))],
    )

    # Disjoint supports: [1, 0, 0] + [0, 0, 1] -> [1, 0, 1] -> / 2.
    merged = tmp_dataset._merge_steps(
        [step_factory(torch.tensor([[0.0], [0.0], [1.0]]))], "mapA"
    )

    assert len(merged) == 1
    assert torch.equal(merged[0]["y_true"], torch.tensor([[0.5], [0.0], [0.5]]))
