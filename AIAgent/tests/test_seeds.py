"""Tests for the training seed helpers and the required config seed field."""

import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from common.config.training_config import TrainingConfig
from ml.dataset import TrainingDataset
from ml.inference import TORCH
from ml.training.seeds import derive_trial_seed, seed_everything
from pydantic import ValidationError


def make_training_config(**overrides) -> dict:
    base = {
        "dynamic_dataset": False,
        "train_percentage": 0.7,
        "threshold_coverage": 100,
        "load_to_cpu": False,
        "epochs": 1,
        "seed": 42,
    }
    base.update(overrides)
    return base


class TestDeriveTrialSeed:
    def test_deterministic(self):
        assert derive_trial_seed(42, 3) == derive_trial_seed(42, 3)

    def test_in_32_bit_range(self):
        for seed in (0, 1, 42, 2**31 - 1):
            for number in range(5):
                assert 0 <= derive_trial_seed(seed, number) < 2**32

    def test_distinct_pairs_give_distinct_seeds(self):
        seeds = {derive_trial_seed(s, n) for s in (0, 42) for n in range(10)}
        assert len(seeds) == 20


class TestSeedEverything:
    def test_python_random_reproducible(self):
        seed_everything(7)
        first = [random.random() for _ in range(10)]
        seed_everything(7)
        second = [random.random() for _ in range(10)]
        assert first == second

    def test_shuffle_reproducible(self):
        items_a, items_b = list(range(50)), list(range(50))
        seed_everything(99)
        random.shuffle(items_a)
        seed_everything(99)
        random.shuffle(items_b)
        assert items_a == items_b

    def test_numpy_reproducible(self):
        seed_everything(3)
        first = np.random.choice([True, False], size=20)
        seed_everything(3)
        second = np.random.choice([True, False], size=20)
        assert (first == second).all()

    def test_torch_reproducible(self):
        seed_everything(11)
        first = torch.rand(4, 4)
        seed_everything(11)
        second = torch.rand(4, 4)
        assert torch.equal(first, second)


class TestTrainingConfigSeed:
    def test_seed_is_required(self):
        data = make_training_config()
        del data["seed"]
        with pytest.raises(ValidationError, match="seed"):
            TrainingConfig(**data)

    def test_valid_config_accepts_seed(self):
        config = TrainingConfig(**make_training_config())
        assert config.seed == 42


def make_processed_dataset(directory: Path, maps: int = 2, steps_per_map: int = 10):
    """A minimal processed dataset: map dirs with a result file and dummy steps."""
    for i in range(maps):
        map_dir = directory / f"map{i}"
        map_dir.mkdir(parents=True)
        (map_dir / "result").write_text("(100, -1, -5, 0)")
        for j in range(steps_per_map):
            (map_dir / f"{j}.pt").write_bytes(b"")
    return directory


def make_similar_steps(count: int) -> list:
    """Steps that are all similar to each other (equal states and vertices)."""
    from torch_geometric.data import HeteroData

    steps = []
    for _ in range(count):
        data = HeteroData()
        data[TORCH.state_vertex].x = torch.zeros(3, 6)
        data[TORCH.game_vertex].x = torch.zeros(4, 7)
        y_true = torch.zeros(3, 1)
        y_true[0] = 1.0
        data["y_true"] = y_true
        steps.append(data)
    return steps


def dedup_pattern(steps: list, kept: list) -> list[int]:
    """Which input positions survived dedup (kept holds references to inputs)."""
    kept_ids = {id(step) for step in kept}
    return [1 if id(step) in kept_ids else 0 for step in steps]


class TestDatasetDeterminism:
    def _build(self, tmp_path: Path, processed_dir: Path, seed: int, **kwargs):
        seed_everything(seed)
        return TrainingDataset(
            raw_dir=tmp_path / "raw",
            processed_dir=processed_dir,
            train_percentage=0.5,
            threshold_coverage=100,
            **kwargs,
        )

    @pytest.fixture
    def processed_dir(self, tmp_path):
        return make_processed_dataset(tmp_path / "processed")

    def test_same_seed_gives_identical_split(self, tmp_path, processed_dir):
        first = self._build(tmp_path, processed_dir, 42)
        second = self._build(tmp_path, processed_dir, 42)
        assert (
            first.train_dataset_indices.indices == second.train_dataset_indices.indices
        )
        assert first.test_dataset_indices.indices == second.test_dataset_indices.indices

    def test_different_seeds_give_different_splits(self, tmp_path, processed_dir):
        first = self._build(tmp_path, processed_dir, 42)
        second = self._build(tmp_path, processed_dir, 43)
        assert (
            first.train_dataset_indices.indices != second.train_dataset_indices.indices
        )

    def test_processed_paths_stable_across_directory_recreation(self, tmp_path):
        # Regression: listdir/glob order changes when a directory is recreated,
        # which used to silently re-shuffle the seeded train/test split.
        processed_dir = make_processed_dataset(tmp_path / "processed")
        first = self._build(tmp_path, processed_dir, 42)
        shutil.rmtree(processed_dir)
        make_processed_dataset(processed_dir)
        second = self._build(tmp_path, processed_dir, 42)
        assert first.processed_paths == second.processed_paths
        assert (
            first.train_dataset_indices.indices == second.train_dataset_indices.indices
        )

    def test_same_seed_gives_identical_step_sampling(self, tmp_path):
        processed_dir = make_processed_dataset(
            tmp_path / "processed", maps=1, steps_per_map=10
        )
        first = self._build(tmp_path, processed_dir, 42, threshold_steps_number=4)
        second = self._build(tmp_path, processed_dir, 42, threshold_steps_number=4)
        assert first.processed_paths == second.processed_paths
        assert len(first.processed_paths) == 4

    def test_same_seed_gives_identical_dedup(self, tmp_path, processed_dir):
        first = self._build(tmp_path, processed_dir, 42, similar_steps_save_prob=0.5)
        second = self._build(tmp_path, processed_dir, 42, similar_steps_save_prob=0.5)
        seed_everything(derive_trial_seed(42, 0))
        pattern_a = dedup_pattern(
            steps_a := make_similar_steps(30), first.remove_similar_steps(steps_a)
        )
        seed_everything(derive_trial_seed(42, 0))
        pattern_b = dedup_pattern(
            steps_b := make_similar_steps(30), second.remove_similar_steps(steps_b)
        )
        assert pattern_a == pattern_b

    def test_different_seeds_give_different_dedup(self, tmp_path, processed_dir):
        first = self._build(tmp_path, processed_dir, 42, similar_steps_save_prob=0.5)
        second = self._build(tmp_path, processed_dir, 43, similar_steps_save_prob=0.5)
        seed_everything(derive_trial_seed(42, 0))
        pattern_a = dedup_pattern(
            steps_a := make_similar_steps(30), first.remove_similar_steps(steps_a)
        )
        seed_everything(derive_trial_seed(43, 0))
        pattern_b = dedup_pattern(
            steps_b := make_similar_steps(30), second.remove_similar_steps(steps_b)
        )
        assert pattern_a != pattern_b
