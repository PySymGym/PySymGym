import json
from pathlib import Path
from typing import List
from common.game import GameMap
from ml.training.dataset import TrainingDataset
from paths import RAW_DATASET_PATH, PROCESSED_DATASET_PATH
from torch_geometric.loader import DataLoader
import torch


def some_train_function(dataset: TrainingDataset, model: torch.nn.Module):

    BATCH_SIZE = 128
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for batch in dataloader:
        ...


def some_val_function(dataset: TrainingDataset, model: torch.nn.Module):
    BATCH_SIZE = 1
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for batch in dataloader:
        ...


def main():
    DATASET_DESCRIPTION = Path("../maps/DotNet/Maps/dataset.json")
    TRAIN_PERCENTAGE = 0.7  # для разделения на train / test

    THRESHOLD_STEPS_NUMBER = None  # эта штука нужна для того, чтобы не брать слишком много шагов с одной карты.
    # Например, если в какой-то карте очень много шагов (больше, чем THRESHOLD_STEPS_NUMBER), то берется рандомное
    # подмножество размером THRESHOLD_STEPS_NUMBER. Если None, то берутся все шаги со всех карт.

    LOAD_TO_CPU = False  # при обучении очередной батч будет подгружаться с диска

    THRESHOLD_COVERAGE = 100  # фильтрация по покрытию
    EPOCHS_NUM = 100

    with open(DATASET_DESCRIPTION, "r") as maps_json:
        maps: List[GameMap] = GameMap.schema().load(
            json.loads(maps_json.read()), many=True
        )

    dataset = TrainingDataset(
        RAW_DATASET_PATH,
        PROCESSED_DATASET_PATH,
        maps,
        train_percentage=TRAIN_PERCENTAGE,
        threshold_steps_number=THRESHOLD_STEPS_NUMBER,
        load_to_cpu=LOAD_TO_CPU,
        threshold_coverage=THRESHOLD_COVERAGE,
    )

    model = ...

    for epoch in range(EPOCHS_NUM):
        dataset.switch_to(
            "train"
        )  # При инициализации датасет делится на train и test. Сейчас мы ему сказали, чтобы он использовал train
        some_train_function(dataset, model)

        # А теперь хотим запустить валидацию
        dataset.switch_to("val")  # говорим датасету, что у нас теперь валидация
        some_val_function(dataset, model)  # и запускаем


if __name__ == "__main__":
    main()
