"""Unit tests for ``ml.validation.loss.validate_loss``.

A real ``TrainingDataset`` (steps written through ``step_factory`` and
``torch.save``) is batched by the production ``DataLoader``; a fixed-output
fake model and a real ``nn.MSELoss`` make the averaged loss exact, and
recorders pin the ``DataLoader``/``tqdm`` kwargs.
"""

from pathlib import Path
from typing import Callable

import pytest
import torch
from ml.dataset import TrainingDataset
from ml.validation.loss import validate_loss as vl
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

pytestmark = pytest.mark.unit


class FixedOutputModel(torch.nn.Module):
    """A model whose output always matches the batched ``y_true`` shape.

    ``infer`` passes the batched node/edge tensors as keyword arguments; the
    output width tracks the batched state-vertex count so a real criterion
    can consume it for any ``batch_size``.
    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def forward(self, state_x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full((state_x.size(0), 1), self._value)


class RecordingDataLoader(DataLoader):
    """A ``DataLoader`` that records the kwargs it was built with."""

    def __init__(self, dataset, **kwargs) -> None:
        self.init_kwargs = kwargs
        super().__init__(dataset, **kwargs)


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


def _dataset(tmp_path: Path, steps: list[HeteroData]) -> TrainingDataset:
    processed = tmp_path / "processed"
    _save_processed_map(processed, "mapA", steps)
    return TrainingDataset(
        raw_dir=tmp_path / "raw",
        processed_dir=processed,
        train_percentage=1.0,
        n_jobs=1,
    )


def _steps(step_factory) -> list[HeteroData]:
    return [
        step_factory(torch.tensor([[0.5], [0.0]])),
        step_factory(torch.tensor([[0.0], [0.7]])),
    ]


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, list[float], list[dict]]:
    """Patch ``DataLoader`` and ``tqdm`` with recorders; return the records."""
    loaders: dict = {}

    def recording_data_loader(dataset, **kwargs):
        loader = RecordingDataLoader(dataset, **kwargs)
        loaders["loader"] = loader
        return loader

    monkeypatch.setattr(vl, "DataLoader", recording_data_loader)
    tqdm_kwargs: list[dict] = []

    def recording_tqdm(iterable, **kwargs):
        tqdm_kwargs.append(kwargs)
        return iterable

    monkeypatch.setattr(vl.tqdm, "tqdm", recording_tqdm)
    return loaders, tqdm_kwargs


def _recording_criterion(
    per_batch_losses: list[float],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    criterion = torch.nn.MSELoss()

    def recording(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = criterion(out, target)
        per_batch_losses.append(loss.item())
        return loss

    return recording


def test_validate_loss_averages_the_thresholded_per_batch_losses(
    tmp_path: Path,
    step_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path, _steps(step_factory))
    loaders, tqdm_kwargs = _patch_seams(monkeypatch)
    per_batch_losses: list[float] = []

    result = vl.validate_loss(
        FixedOutputModel(0.5),
        dataset,
        _recording_criterion(per_batch_losses),
        batch_size=1,
    )

    # y_true is thresholded to {0, 1} before the criterion: both batches are
    # [[1], [0]] vs [[0.5], [0.5]] -> mean squared error 0.25 each. Without
    # the thresholding the average would be 0.135.
    assert result == pytest.approx(0.25)
    assert per_batch_losses == [pytest.approx(0.25), pytest.approx(0.25)]
    assert loaders["loader"].init_kwargs == {"batch_size": 1, "shuffle": False}
    assert tqdm_kwargs == [{"desc": "test", "ncols": 100, "colour": "#975cdb"}]


def test_validate_loss_forwards_batch_size_and_colour(
    tmp_path: Path,
    step_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path, _steps(step_factory))
    loaders, tqdm_kwargs = _patch_seams(monkeypatch)
    per_batch_losses: list[float] = []

    result = vl.validate_loss(
        FixedOutputModel(0.5),
        dataset,
        _recording_criterion(per_batch_losses),
        batch_size=2,
        progress_bar_colour="red",
    )

    # One batch of four thresholded targets vs four 0.5 outputs -> 0.25.
    assert result == pytest.approx(0.25)
    assert per_batch_losses == [pytest.approx(0.25)]
    assert loaders["loader"].init_kwargs == {"batch_size": 2, "shuffle": False}
    assert tqdm_kwargs == [{"desc": "test", "ncols": 100, "colour": "red"}]
