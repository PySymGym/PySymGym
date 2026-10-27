"""Unit tests for ``TrainingDataset`` processing and maintenance.

``process()`` is exercised through the ``serial_pool`` fixture: forked
``mp.Pool`` workers are invisible to pytest-cov, and the pool is an
orchestration seam, not the behaviour under test. The raw dataset layout
mirrors what the game servers serialize: per map a ``result`` file
(space-separated ``coverage negative_tests negative_steps errors``) and, per
step, ``<id>_gameState`` (JSON ``GameState``) plus ``<id>_movedState``
(decimal state Id).
"""

import json
from pathlib import Path

import pytest
import torch
from ml.dataset import (
    Result,
    TrainingDataset,
    convert_input_to_tensor,
)
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
    assert ds.maps_results == {"mapA": Result(100, -1, -5, 0)}
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
    assert ds.maps_results == {"mapA": Result(100, -1, -5, 0)}
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


def test_double_process_raises_file_exists(tmp_path, gamestate_factory, serial_pool):
    # Pinned current behaviour (finding on #632): the processed result file
    # is opened exclusively, so reprocessing an already-processed map
    # crashes; the supported update path is update_map.
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _populate_raw_map(raw, "mapA", gamestate_factory, {"7": 0})
    processed.mkdir()
    ds = TrainingDataset(
        raw_dir=raw, processed_dir=processed, train_percentage=1.0, n_jobs=1
    )
    # Construction already auto-processed mapA.

    with pytest.raises(FileExistsError):
        ds.process()
