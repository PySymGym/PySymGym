"""Online dataset maintenance during validation.

The :class:`MaintenanceMixin` holds the methods that update an already-built
processed dataset while a validation run plays maps: filtering and
de-duplicating incoming steps, merging them with the stored ones, and saving
the result. It is mixed into ``TrainingDataset`` (see
:mod:`ml.dataset.processing`), which provides the state the methods read:

- ``maps_results`` — the per-map :class:`~common.dataset.Result` cache;
- ``processed_dir`` — the processed dataset root;
- ``threshold_coverage`` — the minimum coverage for a map to store steps;
- ``similar_steps_save_prob`` — probability of keeping a similar step;
- ``_load_to_cpu`` / ``_loaded_to_cpu`` — the in-memory step cache.
"""

import glob
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

import numpy as np
import torch
from common.dataset import Result
from ml.device import get_device
from ml.inference import TORCH
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage


class MaintenanceMixin:
    """The online-maintenance half of :class:`TrainingDataset`."""

    def filter_map_steps(self, map_steps: list[HeteroData]) -> list[HeteroData]:
        filtered_map_steps = []
        for step in map_steps:
            if step["y_true"].size()[0] != 1 and not step["y_true"].isnan().any():
                max_ind = torch.argmax(step["y_true"])
                step["y_true"] = torch.zeros_like(step["y_true"])
                step["y_true"][max_ind] = 1.0
                filtered_map_steps.append(step)
        return filtered_map_steps

    def remove_similar_steps(self, map_steps: list[HeteroData]) -> list[HeteroData]:
        def states_num_and_distributions_are_equal(
            step1: HeteroData, step2: HeteroData
        ):
            if step1["y_true"].size()[0] == step2["y_true"].size()[0]:
                cos_d = 1 - torch.nn.functional.cosine_similarity(
                    step1["y_true"], step2["y_true"], dim=0
                )
                return cos_d < 1e-7
            else:
                return False

        def game_vertices_num_is_equal(step1: HeteroData, step2: HeteroData):
            return (
                step1[TORCH.game_vertex]["x"].size()[0]
                == step2[TORCH.game_vertex]["x"].size()[0]
            )

        def save_similar_step():
            return np.random.choice(
                [True, False],
                p=[
                    self.similar_steps_save_prob,
                    1 - self.similar_steps_save_prob,
                ],
            )

        filtered_map_steps = [map_steps[0]]
        for step in map_steps[1:]:
            previous_step = filtered_map_steps[-1]
            if states_num_and_distributions_are_equal(
                step, previous_step
            ) and game_vertices_num_is_equal(step, previous_step):
                if save_similar_step():
                    filtered_map_steps.append(step)
            else:
                filtered_map_steps.append(step)
        return filtered_map_steps

    def _get_map_steps(self, map_name, device=None) -> list[HeteroData]:
        if device is None:
            device = get_device()
        path_to_map_steps = Path(os.path.join(self.processed_dir, map_name))
        map_steps = []
        all_steps_paths = [
            file_path
            for file_path in sorted(glob.glob(os.path.join(path_to_map_steps, "*")))
            if not file_path.endswith("result")
        ]
        for path in all_steps_paths:
            with torch.serialization.safe_globals(
                [BaseStorage, NodeStorage, EdgeStorage]
            ):
                map_steps.append(torch.load(path, map_location=device))
        return map_steps

    def is_update_map_required(self, map_name: str, map_result: Result):
        if map_name in self.maps_results.keys():
            return (
                self.maps_results[map_name] == map_result
                and map_result.coverage_percent == 100
            ) or (self.maps_results[map_name] < map_result)
        else:
            return True

    def update_map(
        self,
        map_name: str,
        map_result: Result,
        map_steps: list[HeteroData],
    ):
        filtered_map_steps = self.remove_similar_steps(self.filter_map_steps(map_steps))
        if map_name in self.maps_results.keys():
            if (
                self.maps_results[map_name] == map_result
                and map_result.coverage_percent == 100
            ):
                init_steps_num = (
                    len(os.listdir(os.path.join(self.processed_dir, map_name))) - 1
                )
                num_of_steps_to_merge = len(filtered_map_steps)
                filtered_map_steps = self._merge_steps(filtered_map_steps, map_name)
                new_steps_num = len(filtered_map_steps)
                if init_steps_num < new_steps_num:
                    logging.info(
                        f"Steps on map {map_name} were merged with current steps with result {tuple(map_result)}. {num_of_steps_to_merge} + {init_steps_num} -> {new_steps_num}. "
                    )
                else:
                    logging.info(
                        f"Existing results reproduced on map {map_name} with result {tuple(map_result)}. {num_of_steps_to_merge} + {init_steps_num} -> {new_steps_num}. "
                    )
                self.maps_results[map_name] = map_result
                self._save_steps(map_name, filtered_map_steps, map_result)
            if self.maps_results[map_name] < map_result:
                logging.info(
                    f"The model with result = {tuple(self.maps_results[map_name])} was replaced with the model with "
                    f"result = {tuple(map_result)} on the map {map_name}"
                )
                self.maps_results[map_name] = map_result
                self._save_steps(map_name, filtered_map_steps, map_result)
        else:
            self.maps_results[map_name] = map_result
            self._save_steps(map_name, filtered_map_steps, map_result)
            logging.info(
                f"New map with name {map_name} was saved with result {tuple(map_result)}"
            )
        del filtered_map_steps
        del map_name
        del map_result

    def _save_steps(self, map_name: str, steps: List[HeteroData], map_result: Result):
        path_to_map_steps = Path(os.path.join(self.processed_dir, map_name))
        if path_to_map_steps.exists():
            shutil.rmtree(path_to_map_steps)
        os.makedirs(path_to_map_steps)
        if map_result.coverage_percent >= self.threshold_coverage:
            for idx, step in enumerate(steps):
                torch.save(step, os.path.join(path_to_map_steps, str(idx) + ".pt"))
        result_file = open(os.path.join(path_to_map_steps, "result"), mode="x")
        result_file.write(str(map_result.to_tuple()))
        result_file.close()
        if self._load_to_cpu:
            self._loaded_to_cpu[map_name] = steps

    def _merge_steps(self, steps: list[HeteroData], map_name: str):
        merged_steps = []

        def create_dict(
            steps_list: list[HeteroData],
        ) -> DefaultDict[int, list[HeteroData]]:
            steps_dict = defaultdict(list)
            for step in steps_list:
                states_num = step[TORCH.state_vertex].x.shape[0]
                game_v_num = step[TORCH.game_vertex].x.shape[0]
                steps_dict[states_num + game_v_num].append(step)
            return steps_dict

        def flatten_and_sort_hetero_data(
            step: HeteroData,
        ) -> Tuple[np.ndarray, np.ndarray]:
            game_dtype = [
                (f"g{i}", np.float32)
                for i in range(step[TORCH.game_vertex].x.shape[-1])
            ]
            game_v = np.sort(
                step[TORCH.game_vertex].x.cpu().numpy().astype(game_dtype),
                order=list(map(lambda x: x[0], game_dtype)),
            )
            states_dtype = [
                (f"s{i}", np.float32)
                for i in range(step[TORCH.state_vertex].x.shape[-1])
            ]
            states = np.sort(
                step[TORCH.state_vertex].x.cpu().numpy().astype(states_dtype),
                order=list(map(lambda x: x[0], states_dtype)),
            )
            return game_v, states

        new_steps = create_dict(steps)
        old_steps = create_dict(self._get_map_steps(map_name, device="cpu"))

        def graphs_are_equal(new_g_v, old_g_v, new_s_v, old_s_v):
            return np.array_equal(new_g_v, old_g_v) and np.array_equal(new_s_v, old_s_v)

        def merge_distributions(old_distribution, new_distribution):
            if old_distribution.device != new_distribution.device:
                device = "cpu"
                old_distribution = old_distribution.to(device)
                new_distribution = new_distribution.to(device)
            distributions_sum = old_distribution + new_distribution
            distributions_sum[distributions_sum != 0] = 1
            merged_distribution = distributions_sum / torch.sum(distributions_sum)
            return merged_distribution

        for vertex_num in new_steps.keys():
            flattened_old_steps = []
            if vertex_num in old_steps.keys():
                for old_step in old_steps[vertex_num]:
                    flattened_old_steps.append(flatten_and_sort_hetero_data(old_step))
            for new_step in new_steps[vertex_num]:
                new_game_vertices, new_state_vertices = flatten_and_sort_hetero_data(
                    new_step
                )
                should_add_to_dataset = True
                for step_num, (old_game_vertices, old_state_vertices) in enumerate(
                    flattened_old_steps
                ):
                    if graphs_are_equal(
                        new_game_vertices,
                        old_game_vertices,
                        new_state_vertices,
                        old_state_vertices,
                    ):
                        old_steps[vertex_num][step_num].y_true = merge_distributions(
                            old_steps[vertex_num][step_num].y_true, new_step.y_true
                        )
                        should_add_to_dataset = False
                        break
                if should_add_to_dataset:
                    merged_steps.append(new_step)
        merged_steps.extend(sum(old_steps.values(), []))
        return merged_steps
