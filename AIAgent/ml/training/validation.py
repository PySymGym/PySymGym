import multiprocessing as mp
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import tqdm
from config import GeneralConfig
from epochs_statistics.tables import create_pivot_table, table_to_string
from learning.play_game import play_game
from ml.training.dataset import TrainingDataset
from ml.training.paths import TRAINING_RESULTS_PATH
from ml.training.utils import append_to_file
from ml.training.wrapper import TrainingModelWrapper
from torch_geometric.loader import DataLoader


def play_game_task(task):
    maps, dataset, wrapper = task[0], task[1], task[2]
    result = play_game(
        with_predictor=wrapper,
        max_steps=GeneralConfig.MAX_STEPS,
        maps=maps,
        with_dataset=dataset,
    )
    torch.cuda.empty_cache()
    return result


def validate_coverage(
    model: torch.nn.Module,
    epoch: int,
    dataset: TrainingDataset,
    run_name: str,
    progress_bar_colour: str = "#ed95ce",
):
    """
    Evaluate model using symbolic execution engine. It runs in parallel and processes
    number is equal to a constant SERVER_COUNT from GeneralConfig.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    epoch : int
        Epoch number to write result.
    dataset : TrainingDataset
        Dataset object for validation.
    run_name : str
        Unique run name to save result.
    progress_bar_colour : str
        Your favorite colour for progress bar.
    """
    wrapper = TrainingModelWrapper(model)
    tasks = [([map], dataset, wrapper) for map in dataset.maps]
    with mp.Pool(GeneralConfig.SERVER_COUNT) as p:
        all_results = []
        for result in tqdm.tqdm(
            p.imap_unordered(play_game_task, tasks, chunksize=1),
            desc="validation",
            total=len(tasks),
            ncols=100,
            colour=progress_bar_colour,
        ):
            all_results.extend(result)

    print(
        "Average dataset state result",
        np.average(
            list(
                map(
                    lambda dataset_map_result: dataset_map_result[0],
                    dataset.maps_results.values(),
                )
            )
        ),
    )
    average_result = np.average(
        list(
            map(
                lambda map_result: map_result.game_result.actual_coverage_percent,
                all_results,
            )
        )
    )

    table, _, _ = create_pivot_table(
        {wrapper: sorted(all_results, key=lambda x: x.map.MapName)}
    )
    table = table_to_string(table)
    results_table_path = Path(os.path.join(TRAINING_RESULTS_PATH, run_name + ".log"))
    append_to_file(
        results_table_path,
        f"Epoch#{epoch}" + " Average coverage: " + str(average_result) + "\n",
    )
    append_to_file(results_table_path, table + "\n")

    return average_result


def validate_loss(
    model: torch.nn.Module,
    epoch: int,
    dataset: TrainingDataset,
    criterion: Callable,
    progress_bar_colour: str = "#975cdb",
):
    epoch_loss = []
    dataloader = DataLoader(dataset, 1)
    for batch in tqdm.tqdm(
        dataloader, desc="test", ncols=100, colour=progress_bar_colour
    ):
        batch.to(GeneralConfig.DEVICE)
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
        loss: torch.Tensor = criterion(out, batch.y_true)
        epoch_loss.append(loss.item())
    result = np.average(epoch_loss)
    print(f"Epoch {epoch}: {result}")
    return result
