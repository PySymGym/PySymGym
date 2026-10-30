"""The training-dataset package.

Split from the former single ``ml/dataset.py`` module:

- :mod:`ml.dataset.processing` — the ``TrainingDataset`` class (building the
  processed dataset, the ``Dataset`` runtime, the ``load_to_cpu`` cache);
- :mod:`ml.dataset.tensors` — the domain→tensor conversion helpers;
- :mod:`ml.dataset.maintenance` — the online-maintenance mixin.

This module re-exports the public names so that ``from ml.dataset import ...``
keeps working: the dataset vocabulary (``Result``, ``TrainingDatasetMode``)
comes from :mod:`common.dataset` and the graph tensor feature widths from
:mod:`ml.schema` (their canonical homes).
"""

from common.dataset import Result, TrainingDatasetMode
from ml.schema import NUM_CFG_VERTEX_FEATURES, NUM_PC_FEATURES, NUM_STATE_FEATURES

from .maintenance import MaintenanceMixin
from .processing import TrainingDataset, flatten_dict
from .tensors import (
    Step,
    convert_input_to_tensor,
    get_hetero_data,
    remove_extra_attrs,
)

__all__ = [
    "MaintenanceMixin",
    "NUM_CFG_VERTEX_FEATURES",
    "NUM_PC_FEATURES",
    "NUM_STATE_FEATURES",
    "Result",
    "Step",
    "TrainingDataset",
    "TrainingDatasetMode",
    "convert_input_to_tensor",
    "flatten_dict",
    "get_hetero_data",
    "remove_extra_attrs",
]
