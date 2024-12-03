import torch
from common.classes import Map2Result, GameMap2SVM
from functools import wraps
from ml.dataset import TrainingDataset
from ml.training.wrapper import TrainingModelWrapper

from ml.validation.game.play_game_via_server import play_game


def catch_return_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e

    return wrapper


@catch_return_exception
def play_game_task(
    task: tuple[GameMap2SVM, TrainingDataset, TrainingModelWrapper],
) -> Map2Result:
    game_map2svm, dataset, wrapper = task
    result = play_game(
        with_predictor=wrapper,
        game_map2svm=game_map2svm,
        with_dataset=dataset,
    )
    torch.cuda.empty_cache()
    return result
