"""Unit tests for the ``TrainingDataset`` lifecycle.

Covers the in-memory (``load_to_cpu``) and on-disk loading paths, the
train/validation index switching, and the metadata refresh. The
``serial_pool`` fixture drives the auto-processing run triggered at the end
of ``__init__`` in-process (forked ``mp.Pool`` workers are invisible to
pytest-cov).
"""

import json
from pathlib import Path

import pytest
import torch
from ml.dataset import (
    Result,
    TrainingDataset,
    TrainingDatasetMode,
)
from torch_geometric.data import HeteroData

pytestmark = pytest.mark.unit


def _save_processed_map(
    processed_dir: Path,
    map_name: str,
    steps: list[HeteroData],
    result: str = "(100, -1, -5, 0)",
) -> Path:
    map_dir = processed_dir / map_name
    map_dir.mkdir(parents=True)
    (map_dir / "result").write_text(result)
    for idx, step in enumerate(steps):
        torch.save(step, map_dir / f"{idx}.pt")
    return map_dir


def _make_dataset(
    tmp_path: Path,
    processed: Path,
    train_percentage: float = 1.0,
    load_to_cpu: bool = False,
) -> TrainingDataset:
    return TrainingDataset(
        raw_dir=tmp_path / "raw",
        processed_dir=processed,
        train_percentage=train_percentage,
        load_to_cpu=load_to_cpu,
        n_jobs=1,
    )


def test_load_to_cpu_constructor_loads_the_steps(tmp_path, step_factory):
    processed = tmp_path / "processed"
    _save_processed_map(
        processed,
        "mapA",
        [
            step_factory(torch.tensor([[1.0], [0.0]])),
            step_factory(torch.tensor([[0.0], [1.0]])),
        ],
    )
    ds = _make_dataset(tmp_path, processed, load_to_cpu=True)
    ds.switch_to(TrainingDatasetMode.TRAINING)

    assert ds.len() == 2
    assert torch.equal(ds[0]["y_true"], torch.tensor([[1.0], [0.0]]))
    assert torch.equal(ds[1]["y_true"], torch.tensor([[0.0], [1.0]]))


def test_get_loads_from_disk_without_load_to_cpu(tmp_path, step_factory):
    processed = tmp_path / "processed"
    _save_processed_map(processed, "mapA", [step_factory(torch.tensor([[1.0], [0.0]]))])
    ds = _make_dataset(tmp_path, processed)
    ds.switch_to(TrainingDatasetMode.TRAINING)

    assert torch.equal(ds[0]["y_true"], torch.tensor([[1.0], [0.0]]))


def test_switch_to_partitions_the_indices(tmp_path, step_factory):
    processed = tmp_path / "processed"
    for i in range(4):
        _save_processed_map(
            processed, f"map{i}", [step_factory(torch.tensor([[1.0], [0.0]]))]
        )
    ds = _make_dataset(tmp_path, processed, train_percentage=0.5)

    ds.switch_to(TrainingDatasetMode.TRAINING)
    train_indices = list(ds.indices())
    ds.switch_to(TrainingDatasetMode.VALIDATION)
    val_indices = list(ds.indices())

    assert len(train_indices) == 2
    assert sorted(train_indices + val_indices) == [0, 1, 2, 3]
    assert not set(train_indices) & set(val_indices)


def test_switch_to_rejects_an_unknown_mode(tmp_dataset):
    with pytest.raises(
        ValueError, match="mode must be either 'TRAINING' or 'VALIDATION'"
    ):
        tmp_dataset.switch_to("train")


def test_update_meta_data_picks_up_new_maps(tmp_path, step_factory):
    processed = tmp_path / "processed"
    _save_processed_map(processed, "mapA", [step_factory(torch.tensor([[1.0], [0.0]]))])
    ds = _make_dataset(tmp_path, processed)
    assert len(ds.processed_paths) == 1
    assert ds.maps_results == {"mapA": Result(100, -1, -5, 0)}

    _save_processed_map(processed, "mapB", [step_factory(torch.tensor([[0.0], [1.0]]))])
    ds.update_meta_data()

    assert len(ds.processed_paths) == 2
    assert ds.maps_results == {
        "mapA": Result(100, -1, -5, 0),
        "mapB": Result(100, -1, -5, 0),
    }
    ds.switch_to(TrainingDatasetMode.TRAINING)
    assert ds.len() == 2


def _populate_raw_map(raw: Path, gamestate_factory) -> None:
    map_dir = raw / "mapA"
    map_dir.mkdir(parents=True)
    (map_dir / "result").write_text("100 1 5 0")
    (map_dir / "7_gameState").write_text(json.dumps(gamestate_factory().to_dict()))
    (map_dir / "7_movedState").write_text("0")


def test_load_to_cpu_dataset_is_usable_after_auto_processing(
    tmp_path, gamestate_factory, serial_pool
):
    # Regression: the auto-processing run at the end of __init__ used to
    # leave _loaded_to_cpu empty (it was computed before processing), so
    # get() raised IndexError after switch_to.
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _populate_raw_map(raw, gamestate_factory)
    processed.mkdir()

    ds = _make_dataset(tmp_path, processed, load_to_cpu=True)
    ds.switch_to(TrainingDatasetMode.TRAINING)

    assert ds.len() == 1
    assert torch.equal(ds[0]["y_true"], torch.tensor([[1.0], [0.0]]))


def test_len_is_stale_until_the_first_switch(tmp_path, gamestate_factory, serial_pool):
    # Pinned current behaviour (finding on #632): update_meta_data refreshes
    # the split but not the active index set, so len() reports 0 until the
    # first switch_to.
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _populate_raw_map(raw, gamestate_factory)
    processed.mkdir()

    ds = _make_dataset(tmp_path, processed)

    assert ds.len() == 0
    ds.switch_to(TrainingDatasetMode.TRAINING)
    assert ds.len() == 1
