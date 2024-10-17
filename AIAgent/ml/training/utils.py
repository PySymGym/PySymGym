import json
import os
import shutil
import typing as t
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common.classes import Map2Result
from common.game import GameMap
from ml.inference import TORCH
from networkx.algorithms.dominance import immediate_dominators
from networkx.classes import DiGraph
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils.convert import to_networkx


def euclidean_dist(y_pred, y_true):
    if len(y_pred) > 1:
        y_pred_min, _ = torch.min(y_pred, dim=0)
        y_pred_norm = y_pred - y_pred_min

        y_true_min, _ = torch.min(y_true, dim=0)
        y_true_norm = y_true - y_true_min
        return torch.sqrt(torch.sum((y_pred_norm - y_true_norm) ** 2))
    else:
        return 0


def get_model(
    path_to_weights: Path, model_init: t.Callable[[], torch.nn.Module]
) -> torch.nn.Module:
    model = model_init()
    weights = torch.load(path_to_weights)
    weights["lin_last.weight"] = torch.tensor(np.random.random([1, 8]))
    weights["lin_last.bias"] = torch.tensor(np.random.random([1]))
    model.load_state_dict(weights)
    return model


def create_folders_if_necessary(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            os.makedirs(path)


def create_file(file: Path):
    open(file, "w").close()


def append_to_file(file: Path, s: str):
    with open(file, "a") as file:
        file.write(s)


def find_unfinished_maps(log_file_path: Path) -> None:
    server_log = open(log_file_path)
    ports = defaultdict(list)
    for line in reversed(list(server_log)):
        splitted = line.split(" ")
        status, map_name, port = splitted[0], splitted[2], splitted[4]
        if status == "Finish":
            ports[port].append(map_name)
        if status == "Start":
            try:
                ports[port].remove(map_name)
            except ValueError:
                print(map_name)
    server_log.close()


def sync_dataset_with_description(dataset_path: Path, description_path: Path) -> None:
    with open(description_path, "r") as maps_json:
        maps_in_description = list(
            map(
                lambda game_map: game_map.MapName,
                GameMap.schema().load(json.loads(maps_json.read()), many=True),
            )
        )
        for saved_map in os.listdir(dataset_path):
            if saved_map not in maps_in_description:
                shutil.rmtree(os.path.join(dataset_path, saved_map))


def remove_call_return_edges(cfg: Data) -> Data:
    shift = 0
    call_return_edges_types = [1, 2]
    for edge_idx, edge_type in enumerate(cfg.edge_attr):
        if edge_type in call_return_edges_types:
            cfg.edge_index = torch.cat(
                (
                    cfg.edge_index[:, 0 : edge_idx - shift],
                    cfg.edge_index[:, edge_idx + 1 - shift :],
                ),
                dim=1,
            )
            shift += 1
    return cfg


def find_entry_points(cfg: Data) -> list[int]:
    entry_points = []
    for vertex_number in range(cfg.x.size()[0]):
        if vertex_number not in cfg.edge_index[1]:
            entry_points.append(vertex_number)
    return entry_points


def find_dominators_in_cfg(graph: HeteroData) -> HeteroData:
    cfg = Data(
        graph[TORCH.game_vertex].x,
        graph[*TORCH.gamevertex_to_gamevertex].edge_index,
        graph[*TORCH.gamevertex_to_gamevertex].edge_type,
    )
    cfg = remove_call_return_edges(cfg)

    entry_points = find_entry_points(cfg)
    if not entry_points:
        raise ValueError("There is no entry points to find dominators from.")
    dominators_graphs: list[DiGraph] = list()
    networkx_cfg: DiGraph = to_networkx(cfg)
    for start in entry_points:
        dominators_graphs.append(
            DiGraph(immediate_dominators(networkx_cfg, start).items())
        )

    for dominators_graph in dominators_graphs:
        for edge in dominators_graph.edges:
            if edge[0] != edge[1]:
                graph[*TORCH.gamevertex_to_gamevertex].edge_index = torch.cat(
                    (
                        graph[*TORCH.gamevertex_to_gamevertex].edge_index,
                        torch.tensor([[edge[1]], [edge[0]]]),
                    ),
                    dim=1,
                )
                graph[*TORCH.gamevertex_to_gamevertex].edge_type = torch.cat(
                    (
                        graph[*TORCH.gamevertex_to_gamevertex].edge_type,
                        torch.tensor([3]),
                    )
                )


def remove_extra_attrs(step: HeteroData):
    if hasattr(step[TORCH.statevertex_history_gamevertex], "edge_attr"):
        del step[TORCH.statevertex_history_gamevertex].edge_attr
    if hasattr(step[TORCH.gamevertex_to_gamevertex], "edge_attr"):
        del step[TORCH.gamevertex_to_gamevertex].edge_attr
    if hasattr(step, "use_for_train"):
        del step.use_for_train


def avg_by_attr(results, path_to_coverage: str) -> int:
    coverage = np.average(
        list(map(lambda result: getattr(result, path_to_coverage), results))
    )
    return coverage


def convert_to_df(map2result_list: list[Map2Result]) -> pd.DataFrame:
    maps = []
    results = []
    for map2result in map2result_list:
        _map = map2result.map
        map_name = _map.GameMap.MapName
        game_result_str = map2result.game_result.printable(verbose=True)
        maps.append(f"{_map.SVMInfo.name} : {map_name}")
        results.append(game_result_str)

    df = pd.DataFrame(results, columns=["Game result"], index=maps).T

    return df
