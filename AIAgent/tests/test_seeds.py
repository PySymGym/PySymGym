"""Tests for the training seed helpers and the required config seed field."""

import random

import numpy as np
import pytest
import torch
from common.config.training_config import TrainingConfig
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
