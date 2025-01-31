import ast
import glob
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import random
import shutil
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
)

import numpy as np
import torch
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
import tqdm
from common.game import GameState
from config import GeneralConfig
from ml.inference import TORCH
from torch.utils.data import random_split
from torch_geometric.data import Dataset, HeteroData

GAMESTATESUFFIX = "_gameState"
STATESUFFIX = "_statesInfo"
MOVEDSTATESUFFIX = "_movedState"

StateId: TypeAlias = int
StateIndex: TypeAlias = int
StateMap: TypeAlias = Dict[StateId, StateIndex]

VertexId: TypeAlias = int
VertexIndex: TypeAlias = int
VertexMap: TypeAlias = Dict[VertexId, VertexIndex]

MapName: TypeAlias = str
Result = namedtuple(
    "Result",
    [
        "coverage_percent",
        "negative_tests_number",
        "negative_steps_number",
        "errors_number",
    ],
)
StatesDistribution: TypeAlias = torch.tensor


@dataclass
class Step:
    graph: HeteroData
    states_distribution: StatesDistribution


class TrainingDataset(Dataset):
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
            step = torch.load(
                self.processed_paths[idx], map_location=GeneralConfig.DEVICE
            )
        remove_extra_attrs(step)
        return step

    def switch_to(self, mode: Literal["train", "val"]) -> None:
        """
        Switch between training and validation modes

        Parameters
        ----------
        mode : str
            'train' --- use training part;\n
            'val' --- use validation part.
        """
        if mode == "train":
            self.__indices = self.train_dataset_indices
        elif mode == "val":
            self.__indices = self.test_dataset_indices
        else:
            raise ValueError("mode must be either 'train' or 'val'")

    @property
    def raw_dir(self):
        return self._raw_dir

    @property
    def processed_dir(self):
        return self._processed_dir

    def _get_processed_paths(self) -> List[str]:
        all_files = []
        for map_name in os.listdir(self.processed_dir):
            if map_name in self.maps_results:
                if (
                    self.maps_results[map_name].coverage_percent
                    >= self.threshold_coverage
                ):
                    all_steps_paths = [
                        file_path
                        for file_path in glob.glob(
                            os.path.join(self.processed_dir, map_name, "*.pt")
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
        self.maps_results = self._get_results()
        self._processed_paths = self._get_processed_paths()
        self.train_dataset_indices, self.test_dataset_indices = self._split_dataset()
        if self._load_to_cpu:
            self._flattened_loaded_steps = flatten_dict(self._loaded_to_cpu)

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def len(self):
        return len(self.__indices)

    def _get_results(self) -> Dict[str, Result]:
        results = dict()
        for map_name in os.listdir(self.processed_dir):
            path_to_map_steps = Path(
                os.path.join(self.processed_dir, map_name, "result")
            )
            if path_to_map_steps.exists():
                result_file = open(path_to_map_steps)
                result = Result(*ast.literal_eval(result_file.read()))
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
                negative_tests_number=-result[1],
                negative_steps_number=-result[2],
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
                os.listdir(self.raw_dir),
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
                is_successful: list[bool] = list(
                    p.map(process_and_save_step_task, tasks)
                )
                if len(is_successful) != len(raw_ids):
                    logging.warning(f"Processing of map {map_name} failed somewhere.")
                result = process_result(map_name)
                result_file = open(os.path.join(processed_map_path, "result"), mode="x")
                result_file.write(str(tuple(result)))
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
            steps[map_name].append(torch.load(step_path, map_location="cpu"))
        return steps

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

    def _get_map_steps(self, map_name) -> list[HeteroData]:
        path_to_map_steps = Path(os.path.join(self.processed_dir, map_name))
        map_steps = []
        all_steps_paths = [
            file_path
            for file_path in glob.glob(os.path.join(path_to_map_steps, "*"))
            if not file_path.endswith("result")
        ]
        for path in all_steps_paths:
            with torch.serialization.safe_globals(
                [BaseStorage, NodeStorage, EdgeStorage]
            ):
                map_steps.append(torch.load(path, map_location=GeneralConfig.DEVICE))
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
        if map_result[0] >= self.threshold_coverage:
            for idx, step in enumerate(steps):
                torch.save(step, os.path.join(path_to_map_steps, str(idx) + ".pt"))
        result_file = open(os.path.join(path_to_map_steps, "result"), mode="x")
        result_file.write(str(tuple(map_result)))
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
        old_steps = create_dict(self._get_map_steps(map_name))

        def graphs_are_equal(new_g_v, old_g_v, new_s_v, old_s_v):
            return np.array_equal(new_g_v, old_g_v) and np.array_equal(new_s_v, old_s_v)

        def merge_distributions(old_distribution, new_distribution):
            if old_distribution.device != new_distribution.device:
                device = GeneralConfig.DEVICE
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


def flatten_dict(dict_to_flatten: Dict) -> List[Any]:
    return sum(dict_to_flatten.values(), [])


def convert_input_to_tensor(
    input: GameState,
) -> Tuple[HeteroData, Dict[StateId, StateIndex]]:
    """
    Converts game env to tensors
    """
    graphVertices = input.GraphVertices
    game_states, game_edges = input.States, input.Map
    data = HeteroData()
    nodes_vertex, edges_index_v_v, edges_attr_v_v, edges_types_v_v = [], [], [], []
    nodes_state, edges_index_s_s = [], []
    edges_index_s_v_in, edges_index_v_s_in = [], []
    edges_index_s_v_history, edges_index_v_s_history = [], []
    edges_attr_s_v, edges_attr_v_s = [], []

    state_map: Dict[StateId, StateIndex] = dict()
    vertex_map: Dict[VertexId, VertexIndex] = dict()
    vertex_index = 0
    state_index = 0

    # vertex nodes
    for v in graphVertices:
        vertex_id = v.Id
        if vertex_id not in vertex_map:
            vertex_map[vertex_id] = vertex_index  # maintain order in tensors
            vertex_index = vertex_index + 1
            nodes_vertex.append(
                np.array(
                    [
                        int(v.InCoverageZone),
                        v.BasicBlockSize,
                        int(v.CoveredByTest),
                        int(v.VisitedByState),
                        int(v.TouchedByState),
                        int(v.ContainsCall),
                        int(v.ContainsThrow),
                    ]
                )
            )
    # vertex -> vertex edges
    for e in game_edges:
        edges_index_v_v.append(
            np.array([vertex_map[e.VertexFrom], vertex_map[e.VertexTo]])
        )
        edges_attr_v_v.append(np.array([e.Label.Token]))
        edges_types_v_v.append(e.Label.Token)

    state_doubles = 0

    # state nodes
    for s in game_states:
        state_id = s.Id
        if state_id not in state_map:
            state_map[state_id] = state_index
            nodes_state.append(
                np.array(
                    [
                        s.Position,
                        s.PathConditionSize,
                        s.VisitedAgainVertices,
                        s.VisitedNotCoveredVerticesInZone,
                        s.VisitedNotCoveredVerticesOutOfZone,
                        s.StepWhenMovedLastTime,
                        s.InstructionsVisitedInCurrentBlock,
                    ]
                )
            )
            # history edges: state -> vertex and back
            for h in s.History:
                v_to = vertex_map[h.GraphVertexId]
                edges_index_s_v_history.append(np.array([state_index, v_to]))
                edges_index_v_s_history.append(np.array([v_to, state_index]))
                edges_attr_s_v.append(
                    np.array([h.NumOfVisits, h.StepWhenVisitedLastTime])
                )
                edges_attr_v_s.append(
                    np.array([h.NumOfVisits, h.StepWhenVisitedLastTime])
                )
            state_index = state_index + 1
        else:
            state_doubles += 1

    # state and its childen edges: state -> state
    for s in game_states:
        for ch in s.Children:
            try:
                edges_index_s_s.append(np.array([state_map[s.Id], state_map[ch]]))
            except KeyError:
                print("[ERROR]: KeyError")
                return None, None

    # state position edges: vertex -> state and back
    for v in graphVertices:
        for s in v.States:
            edges_index_s_v_in.append(np.array([state_map[s], vertex_map[v.Id]]))
            edges_index_v_s_in.append(np.array([vertex_map[v.Id], state_map[s]]))

    data[TORCH.game_vertex].x = torch.tensor(np.array(nodes_vertex), dtype=torch.float)
    data[TORCH.state_vertex].x = torch.tensor(np.array(nodes_state), dtype=torch.float)

    def tensor_not_empty(tensor):
        return tensor.numel() != 0

    # dumb fix
    def null_if_empty(tensor):
        return (
            tensor
            if tensor_not_empty(tensor)
            else torch.empty((2, 0), dtype=torch.int64)
        )

    data[*TORCH.gamevertex_to_gamevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_v_v), dtype=torch.long).t().contiguous()
    )
    data[*TORCH.gamevertex_to_gamevertex].edge_attr = torch.tensor(
        np.array(edges_attr_v_v), dtype=torch.long
    )
    data[*TORCH.gamevertex_to_gamevertex].edge_type = torch.tensor(
        np.array(edges_types_v_v), dtype=torch.long
    )
    data[*TORCH.statevertex_in_gamevertex].edge_index = (
        torch.tensor(np.array(edges_index_s_v_in), dtype=torch.long).t().contiguous()
    )
    data[*TORCH.gamevertex_in_statevertex].edge_index = (
        torch.tensor(np.array(edges_index_v_s_in), dtype=torch.long).t().contiguous()
    )
    data[*TORCH.statevertex_history_gamevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_s_v_history), dtype=torch.long)
        .t()
        .contiguous()
    )
    data[*TORCH.gamevertex_history_statevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_v_s_history), dtype=torch.long)
        .t()
        .contiguous()
    )
    data[*TORCH.gamevertex_history_statevertex].edge_attr = torch.tensor(
        np.array(edges_attr_s_v), dtype=torch.long
    )
    data[*TORCH.gamevertex_history_statevertex].edge_attr = torch.tensor(
        np.array(edges_attr_v_s), dtype=torch.long
    )
    # if (edges_index_s_s): #TODO: empty?
    data[*TORCH.statevertex_parentof_statevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_s_s), dtype=torch.long).t().contiguous()
    )
    return data, state_map


def remove_extra_attrs(step: HeteroData):
    if hasattr(step[TORCH.statevertex_history_gamevertex], "edge_attr"):
        del step[TORCH.statevertex_history_gamevertex].edge_attr
    if hasattr(step[TORCH.gamevertex_to_gamevertex], "edge_attr"):
        del step[TORCH.gamevertex_to_gamevertex].edge_attr
    if hasattr(step, "use_for_train"):
        del step.use_for_train
