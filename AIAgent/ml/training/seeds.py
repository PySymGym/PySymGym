"""Deterministic seeding for training runs.

A run is reproducible when every random source it draws from is seeded: the
dataset split and step sampling (``torch`` / ``random``), similar-step dedup
(``numpy``), the validation map order (``random``), and model weight
initialization (``torch``). The study seed covers everything done before a
trial starts; each trial then re-seeds all sources from a seed derived from
the study seed and its own number.
"""

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every global RNG the training pipeline draws from.

    Parameters
    ----------
    seed : int
        The seed to apply to ``random``, ``numpy``, and ``torch`` (CPU and
        all CUDA devices).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_trial_seed(study_seed: int, trial_number: int) -> int:
    """Derive the seed of one trial from the study seed and the trial number.

    The derivation is a SHA-256 digest of ``"<study_seed>:<trial_number>"``
    truncated to 32 bits, so it is deterministic across platforms and runs
    (unlike Python's salted ``hash``) and well distributed for any pair of
    inputs.

    Parameters
    ----------
    study_seed : int
        The seed from the training config.
    trial_number : int
        The Optuna trial number.

    Returns
    -------
    int
        A seed in ``[0, 2**32)``.
    """
    digest = hashlib.sha256(f"{study_seed}:{trial_number}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")
