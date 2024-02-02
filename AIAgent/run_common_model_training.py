import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import typing as t
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
import tqdm
from common.game import GameMap
from config import GeneralConfig
from epochs_statistics.tables import create_pivot_table, table_to_string
from learning.play_game import play_game
from ml.common_model.dataset import FullDataset
from ml.common_model.paths import (
    BEST_MODELS_DICT_PATH,
    COMMON_MODELS_PATH,
    DATASET_MAP_RESULTS_FILENAME,
    DATASET_ROOT_PATH,
    PRETRAINED_MODEL_PATH,
    TRAINING_DATA_PATH,
    RAW_FILES_PATH,
)
from ml.common_model.utils import csv2best_models, get_model
from ml.common_model.wrapper import BestModelsWrapper, CommonModelWrapper
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2.model_modified import (
    StateModelEncoderLastLayer,
)
from ml.models.StateGNNEncoderConvEdgeAttr.model_modified import (
    StateModelEncoderLastLayer as RefStateModelEncoderLastLayer,
)
from ml.data_loader_compact import ServerDataloaderHeteroVector
from torch_geometric.loader import DataLoader
from ml.data_loader_compact import ServerDataloaderHeteroVector
import optuna
from functools import partial
import joblib

LOG_PATH = Path("./ml_app.log")
TABLES_PATH = Path("./ml_tables.log")
COMMON_MODELS_PATH = Path(COMMON_MODELS_PATH)
BEST_MODELS_DICT = Path(BEST_MODELS_DICT_PATH)
TRAINING_DATA_PATH = Path(TRAINING_DATA_PATH)
DATASET_ROOT_PATH = Path(DATASET_ROOT_PATH)


logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)

if not COMMON_MODELS_PATH.exists():
    os.makedirs(COMMON_MODELS_PATH)

if not BEST_MODELS_DICT.exists():
    os.makedirs(BEST_MODELS_DICT_PATH)

if not TRAINING_DATA_PATH.exists():
    os.makedirs(TRAINING_DATA_PATH)

if not DATASET_ROOT_PATH.exists():
    os.makedirs(DATASET_ROOT_PATH)


def create_file(file: Path):
    open(file, "w").close()


def append_to_file(file: Path, s: str):
    with open(file, "a") as file:
        file.write(s)


def play_game_task(task):
    maps, dataset, cmwrapper = task[0], task[1], task[2]
    result = play_game(
        with_predictor=cmwrapper,
        max_steps=GeneralConfig.MAX_STEPS,
        maps=maps,
        with_dataset=dataset,
    )
    return result


@dataclass
class TrialSettings:
    lr: float
    epochs: int
    batch_size: int
    optimizer: torch.optim.Optimizer
    loss: any
    random_seed: int


def train(trial: optuna.trial.Trial, dataset: FullDataset, maps: list[GameMap]):
    config = TrialSettings(
        lr=trial.suggest_float("lr", 1e-7, 1e-3),
        batch_size=trial.suggest_int("batch_size", 32, 1024),
        epochs=10,
        optimizer=trial.suggest_categorical("optimizer", [torch.optim.Adam]),
        loss=trial.suggest_categorical("loss", [nn.KLDivLoss]),
        random_seed=937,
    )
    np.random.seed(config.random_seed)
    # for name, param in model.named_parameters():
    #     if "lin_last" not in name:
    #         param.requires_grad = False

    path_to_weights = os.path.join(
        PRETRAINED_MODEL_PATH,
        "RGCNEdgeTypeTAG3VerticesDoubleHistory2",
        "64ch",
        "20e",
        "GNN_state_pred_het_dict",
    )
    model = get_model(
        Path(path_to_weights),
        lambda: StateModelEncoderLastLayer(hidden_channels=64, out_channels=8),
    )

    model.to(GeneralConfig.DEVICE)
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    criterion = config.loss()

    timestamp = datetime.now().timestamp()
    run_name = (
        f"{datetime.fromtimestamp(timestamp)}_{config.batch_size}_Adam_{config.lr}_KLDL"
    )

    print(run_name)
    path_to_saved_models = os.path.join(COMMON_MODELS_PATH, run_name)
    os.makedirs(path_to_saved_models)
    TABLES_PATH = Path(os.path.join(TRAINING_DATA_PATH, run_name + ".log"))
    create_file(TABLES_PATH)
    create_file(LOG_PATH)

    cmwrapper = CommonModelWrapper(model)

    tasks = [([map], FullDataset("", ""), cmwrapper) for map in maps]

    mp.set_start_method("spawn", force=True)

    all_average_results = []
    for epoch in range(config.epochs):
        data_list = dataset.get_plain_data(map_result_threshold=80)
        data_loader = DataLoader(data_list, batch_size=config.batch_size, shuffle=False)
        print("DataLoader size", len(data_loader))

        model.train()
        for batch in tqdm.tqdm(data_loader, desc="training"):
            batch.to(GeneralConfig.DEVICE)
            optimizer.zero_grad()

            out = model(
                game_x=batch["game_vertex"].x,
                state_x=batch["state_vertex"].x,
                edge_index_v_v=batch["game_vertex", "to", "game_vertex"].edge_index,
                edge_type_v_v=batch["game_vertex", "to", "game_vertex"].edge_type,
                edge_index_history_v_s=batch[
                    "game_vertex", "history", "state_vertex"
                ].edge_index,
                edge_attr_history_v_s=batch[
                    "game_vertex", "history", "state_vertex"
                ].edge_attr,
                edge_index_in_v_s=batch["game_vertex", "in", "state_vertex"].edge_index,
                edge_index_s_s=batch[
                    "state_vertex", "parent_of", "state_vertex"
                ].edge_index,
            )
            y_true = batch.y_true
            loss = criterion(out, y_true)
            if loss != 0:
                loss.backward()
                optimizer.step()
            del out
            del batch
            torch.cuda.empty_cache()

        # validation
        model.eval()
        cmwrapper.make_copy(str(epoch + 1))

        with mp.Pool(GeneralConfig.SERVER_COUNT) as p:
            result = list(p.map(play_game_task, tasks, chunksize=1))

            all_results = []
            for maps_result, maps_data in result:
                for map_name in maps_data.keys():
                    dataset.update(
                        map_name, maps_data[map_name][0], maps_data[map_name][1], True
                    )
                all_results += maps_result

            dataset.save()

        print(
            "Average dataset_state result",
            np.average(list(map(lambda x: x[0][0], dataset.maps_data.values()))),
        )
        average_result = np.average(
            list(map(lambda x: x.game_result.actual_coverage_percent, all_results))
        )
        all_average_results.append(average_result)
        table, _, _ = create_pivot_table(
            {cmwrapper: sorted(all_results, key=lambda x: x.map.MapName)}
        )
        table = table_to_string(table)
        append_to_file(
            TABLES_PATH,
            f"Epoch#{epoch}" + " Average coverage: " + str(average_result) + "\n",
        )
        append_to_file(TABLES_PATH, table + "\n")

        path_to_model = os.path.join(COMMON_MODELS_PATH, run_name, str(epoch + 1))
        torch.save(model.state_dict(), Path(path_to_model))
        del data_list
        del data_loader

    return max(all_average_results)


def generate_dataset():
    dataset = FullDataset(DATASET_ROOT_PATH, DATASET_MAP_RESULTS_FILENAME)
    loader = ServerDataloaderHeteroVector(Path(RAW_FILES_PATH), DATASET_ROOT_PATH)
    loader.save_dataset_for_training(
        DATASET_MAP_RESULTS_FILENAME, num_processes=mp.cpu_count() - 1
    )
    dataset.load()
    return dataset


def get_dataset():
    dataset = FullDataset(DATASET_ROOT_PATH, DATASET_MAP_RESULTS_FILENAME)
    dataset.load()
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasetdescription",
        type=str,
        help="full paths to JSON-file with dataset description",
        required=True,
    )
    parser.add_argument(
        "--datasetbasepath",
        type=str,
        help="path to dir with explored dlls",
        required=True,
    )
    parser.add_argument(
        "--generatedataset",
        type=bool,
        help="set this flag if dataset generation is needed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    dataset_base_path = args.datasetbasepath

    print(GeneralConfig.DEVICE)

    with open(args.datasetdescription, "r") as maps_json:
        maps = GameMap.schema().load(json.loads(maps_json.read()), many=True)
        for map in maps:
            fullName = os.path.join(dataset_base_path, map.AssemblyFullName)
            map.AssemblyFullName = fullName

    if args.generatedataset:
        dataset = generate_dataset()
    else:
        dataset = get_dataset()

    print(GeneralConfig.DEVICE)

    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    objective = partial(train, dataset=dataset, maps=maps)
    study.optimize(objective, n_trials=100)
    joblib.dump(study, f"{datetime.fromtimestamp(datetime.now().timestamp())}.pkl")


if __name__ == "__main__":
    main()
