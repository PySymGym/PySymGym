import json
import os
import shutil
import typing as t
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

from common.game import GameMap


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
