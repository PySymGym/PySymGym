"""Building the raw→processed dataset and the ``TrainingDataset`` runtime.

:class:`TrainingDataset` owns the dataset lifecycle: locating the raw and
processed files, building the processed dataset with a process pool
(:meth:`TrainingDataset.process`), the PyTorch ``Dataset`` runtime
(splitting, indexing, loading) and the in-memory ``load_to_cpu`` cache. The
online-maintenance methods are mixed in from
:mod:`ml.dataset.maintenance`; the domain→tensor conversion helpers live in
:mod:`ml.dataset.tensors`.
"""

import ast
import glob
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import tqdm
from common.dataset import Result, TrainingDatasetMode
from common.game import GameState
from ml.device import get_device
from ml.inference import degenerate_counts
from torch.utils.data import random_split
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage

from .maintenance import MaintenanceMixin
from .tensors import (
    MapName,
    StateMap,
    StatesDistribution,
    Step,
    convert_input_to_tensor,
)

GAMESTATESUFFIX = "_gameState"
STATESUFFIX = "_statesInfo"
MOVEDSTATESUFFIX = "_movedState"


class TrainingDataset(MaintenanceMixin, Dataset):
    """
    Dataset class which have information about data location and functions for filtering, updating and saving it.

    Parameters
    ----------
    raw_dir : Path
        Path to raw files that need to be processed.
    processed_dir : Path
        Path to processed dataset.
    train_percentage : float
        Percentage of training dataset. (1 - train_percentage) * full_dataset = validation_dataset
    threshold_steps_number : int
        If number of map's saved steps > threshold_steps_number take sample from saved steps.
    load_to_cpu : bool, optional
        If set to True, dataset will be loaded to cpu. Be careful with this because dataset can be really large.
    threshold_coverage : int, optional
        Use only maps with coverage which greater or equal then threshold_coverage for training.
    similar_steps_save_prob : float, optional
        Probability to save similar steps to dataset with.
    progress_bar_color : str, optional
        The color of progress bar during processing and dataset loading :)
    n_jobs: int, optional
        Number of parallel processes to use to process dataset.
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        train_percentage: float,
        threshold_steps_number: Optional[int] = None,
        load_to_cpu: bool = False,
        threshold_coverage: int = 100,
        similar_steps_save_prob=0,
        progress_bar_color: str = "#975cdb",
        n_jobs: int = mp.cpu_count() - 1,
    ):
        self.n_jobs = n_jobs
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir

        self.threshold_coverage = threshold_coverage
        self.threshold_steps_number = threshold_steps_number
        self.maps_results = self._get_results()
        self.similar_steps_save_prob = similar_steps_save_prob
        self._progress_bar_color = progress_bar_color
        self._processed_paths = self._get_processed_paths()
        self._load_to_cpu = load_to_cpu
        if self._load_to_cpu:
            self._loaded_to_cpu = self._load_steps()
            self._flattened_loaded_steps = flatten_dict(self._loaded_to_cpu)

        self.train_percentage = train_percentage
        self.train_dataset_indices, self.test_dataset_indices = self._split_dataset()
        self.__indices = self.train_dataset_indices
        super().__init__()

    def _split_dataset(self) -> Tuple[List, List]:
        full_dataset_len = len(self.processed_paths)
        train_dataset_len = round(full_dataset_len * self.train_percentage)
        train_dataset_indices, test_dataset_indices = random_split(
            range(full_dataset_len),
            [train_dataset_len, full_dataset_len - train_dataset_len],
        )
        return train_dataset_indices, test_dataset_indices

    def indices(self) -> Sequence:
        return self.__indices

    def get(self, idx) -> HeteroData:
        """
        Get item with given index.
        """
        if self._load_to_cpu:
            step = self._flattened_loaded_steps[idx]
        else:
            with torch.serialization.safe_globals(
                [BaseStorage, NodeStorage, EdgeStorage]
            ):
                step = torch.load(self.processed_paths[idx], map_location=get_device())
        return step

    def switch_to(self, mode: TrainingDatasetMode) -> None:
        """
        Switch between training and validation modes

        Parameters
        ----------
        mode : str
            'train' --- use training part;\n
            'val' --- use validation part.
        """
        if mode is TrainingDatasetMode.TRAINING:
            self.__indices = self.train_dataset_indices
        elif mode is TrainingDatasetMode.VALIDATION:
            self.__indices = self.test_dataset_indices
        else:
            raise ValueError("mode must be either 'TRAINING' or 'VALIDATION'")

    @property
    def raw_dir(self):
        return self._raw_dir

    @property
    def processed_dir(self):
        return self._processed_dir

    def _get_processed_paths(self) -> List[str]:
        all_files = []
        # Sorted enumeration: listdir/glob order changes when the directory is
        # recreated, which would silently re-shuffle the seeded train/test split.
        for map_name in sorted(os.listdir(self.processed_dir)):
            if map_name in self.maps_results:
                if (
                    self.maps_results[map_name].coverage_percent
                    >= self.threshold_coverage
                ):
                    all_steps_paths = [
                        file_path
                        for file_path in sorted(
                            glob.glob(
                                os.path.join(self.processed_dir, map_name, "*.pt")
                            )
                        )
                    ]
                    if (self.threshold_steps_number is None) or len(
                        all_steps_paths
                    ) <= self.threshold_steps_number:
                        all_files.extend(all_steps_paths)
                    else:
                        all_files.extend(
                            random.sample(all_steps_paths, self.threshold_steps_number)
                        )
        return all_files

    def update_meta_data(self) -> None:
        """Refresh the cached metadata from the processed directory.

        Re-reads the per-map ``result`` files, the processed step paths and
        the train/test split. In ``load_to_cpu`` mode the in-memory steps are
        reloaded from disk as well, so they stay consistent with the files
        written by ``process``/``_save_steps`` — including the steps created
        by the auto-processing run triggered at the end of ``__init__``.

        Note: the active index set (``indices()``) is not touched; call
        :meth:`switch_to` to re-select it.
        """
        self.maps_results = self._get_results()
        self._processed_paths = self._get_processed_paths()
        self.train_dataset_indices, self.test_dataset_indices = self._split_dataset()
        if self._load_to_cpu:
            self._loaded_to_cpu = self._load_steps()
            self._flattened_loaded_steps = flatten_dict(self._loaded_to_cpu)

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def len(self):
        return len(self.__indices)

    def _get_results(self) -> Dict[str, Result]:
        results = dict()
        for map_name in sorted(os.listdir(self.processed_dir)):
            path_to_map_steps = Path(
                os.path.join(self.processed_dir, map_name, "result")
            )
            if path_to_map_steps.exists():
                result_file = open(path_to_map_steps)
                result = Result.from_tuple(ast.literal_eval(result_file.read()))
                results[map_name] = result
                result_file.close()
        return results

    def process(self):
        def process_result(map_name: MapName) -> Result:
            f = open(os.path.join(self.raw_dir, map_name, "result"))
            result = f.read()
            f.close()
            result = tuple(map(lambda x: int(x), result.split()))
            return Result(
                coverage_percent=result[0],
                tests_number=result[1],
                steps_number=result[2],
                errors_number=result[3],
            )

        def get_step_raw_ids(map_name: str) -> List[str]:
            file_names = os.listdir(os.path.join(self.raw_dir, map_name))
            step_ids = list(
                set(map(lambda file_name: file_name.split("_")[0], file_names))
            )
            step_ids.remove("result")
            return step_ids

        with mp.Pool(self.n_jobs) as p:
            for map_name in tqdm.tqdm(
                sorted(os.listdir(self.raw_dir)),
                desc="Dataset processing",
                ncols=100,
                colour=self._progress_bar_color,
            ):
                raw_map_path = Path(self.raw_dir / map_name)
                processed_map_path = Path(self.processed_dir / map_name)
                if not processed_map_path.exists():
                    os.makedirs(processed_map_path)

                process_and_save_step_task = partial(
                    self.process_and_save_step, raw_map_path, processed_map_path
                )
                raw_ids = get_step_raw_ids(map_name)
                tasks = list(
                    (raw_step_id, processed_step_id)
                    for processed_step_id, raw_step_id in enumerate(sorted(raw_ids))
                )
                # p.map re-raises worker exceptions, so a step that fails to
                # convert aborts the run instead of leaving a partial map.
                p.map(process_and_save_step_task, tasks)
                result = process_result(map_name)
                # mode="w": re-running process() over an already-processed
                # map overwrites the result file (the .pt steps are already
                # overwritten in place by torch.save), so the re-run is
                # idempotent.
                result_file = open(os.path.join(processed_map_path, "result"), mode="w")
                result_file.write(str(result.to_tuple()))
                result_file.close()
        self.update_meta_data()

    def process_and_save_step(
        self, raw_map_path: Path, processed_map_path: Path, ids: tuple[str, int]
    ) -> bool:
        def get_states_distribution(
            file_path: str, state_map: StateMap
        ) -> StatesDistribution:
            states_distribution = torch.zeros([len(state_map.keys()), 1])
            f = open(file_path)
            state_id = int(f.read())
            f.close()
            states_distribution[state_map[state_id]] = 1
            return states_distribution

        raw_step_id, processed_step_id = ids
        f = open(os.path.join(raw_map_path, raw_step_id + GAMESTATESUFFIX))
        data = json.load(f)
        f.close()
        graph, state_map = convert_input_to_tensor(GameState.from_dict(data))
        # Degenerate steps (#619) are skipped before get_states_distribution:
        # with an empty state_map the moved-state lookup would KeyError.
        # Zero-path-condition graphs are not degenerate and are saved as usual.
        counts = degenerate_counts(graph)
        if counts is not None:
            logging.warning(
                "skipping degenerate step %s/%s: %d states, %d game vertices",
                raw_map_path.name,
                raw_step_id,
                *counts,
            )
            return False
        distribution = get_states_distribution(
            os.path.join(raw_map_path, raw_step_id + MOVEDSTATESUFFIX), state_map
        )
        step = Step(graph, distribution)
        step.graph.y_true = step.states_distribution
        torch.save(
            step.graph,
            os.path.join(processed_map_path, str(processed_step_id) + ".pt"),
        )
        return True

    def _load_steps(self):
        def get_map_name_from_path(path: Path):
            return osp.split(osp.split(path)[0])[1]

        steps = defaultdict(list)
        for step_path in tqdm.tqdm(
            self.processed_paths,
            desc="Dataset loading",
            ncols=100,
            colour=self._progress_bar_color,
        ):
            map_name = get_map_name_from_path(step_path)
            with torch.serialization.safe_globals(
                [BaseStorage, NodeStorage, EdgeStorage]
            ):
                steps[map_name].append(torch.load(step_path, map_location="cpu"))
        return steps


def flatten_dict(dict_to_flatten: Dict) -> List[Any]:
    return sum(dict_to_flatten.values(), [])
