"""Dataset vocabulary shared across the agent packages.

Dependency-light on purpose: the shared layer (``common``) must stay
importable without the ML stack, so the dataset types it defines live
here, not in ``ml.dataset``.
"""

from enum import Enum


class TrainingDatasetMode(Enum):
    """Which split of the training dataset is active.

    The member *values* are serialized into configs and MLflow artifacts,
    so they are pinned behaviour, not implementation detail.
    """

    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
