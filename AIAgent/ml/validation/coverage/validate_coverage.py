import logging
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from typing import Optional

import torch
import tqdm
from common.classes import GameResult, Map2Result
from common.config.validation_config import (
    SVMValidation,
    SVMValidationSendEachStep,
    SVMValidationSendModel,
)
from common.game import GameMap2SVM
from ml.dataset import Result, TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from ml.validation.coverage.game_managers.base_game_manager import BaseGameManager
from ml.validation.coverage.game_managers.each_step.each_step_game_manager import (
    EachStepGameManager,
)
from ml.validation.coverage.game_managers.model.process_game_manager import (
    ModelGameManager,
)
from ml.validation.coverage.validate_coverage_utils import catch_return_exception


class ValidationCoverage:
    """
    Performs coverage validation of a model using symbolic execution. This class manages the parallel execution
    of game simulations across multiple maps and updates a provided dataset with the results.

    Attributes:
        model (`torch.nn.Module`): The model to be validated.
        dataset (`Optional[TrainingDataset]`): The dataset to update with validation results. Can be `None` if dataset update is not required.
    """

    def __init__(self, model: torch.nn.Module, dataset: Optional[TrainingDataset]):
        self.model = model
        self.dataset = dataset
        self._game_manager: Optional[BaseGameManager] = None

    def _evaluate_game_map(
        self,
        game_map2svm: GameMap2SVM,
    ) -> Map2Result | Exception:
        if self._game_manager is None:
            raise RuntimeError("Game manager has not been initialized yet")
        catching_play_game_map = catch_return_exception(
            self._game_manager.play_game_map
        )
        result: Map2Result | Exception = catching_play_game_map(game_map2svm)
        game_map = game_map2svm.GameMap
        if (
            self.dataset is not None
            and isinstance(result, Map2Result)
            and isinstance(result.game_result, GameResult)
        ):
            game_result = result.game_result
            map_result = Result(
                coverage_percent=game_result.actual_coverage_percent,
                negative_tests_number=-game_result.tests_count,
                negative_steps_number=-game_result.steps_count,
                errors_number=game_result.errors_count,
            )
            map_name = game_map.MapName
            if self.dataset.is_update_map_required(map_name, map_result):
                steps = self._game_manager.get_game_steps(game_map)
                if steps is not None:
                    self.dataset.update_map(map_name, map_result, steps)
                else:
                    logging.debug(f"Failed to obtain steps of game={str(game_map2svm)}")
                del steps
        elif isinstance(result, Exception):
            logging.error(result, exc_info=True)
        self._game_manager.delete_game_artifacts(game_map)
        return result

    def validate_coverage(
        self,
        maps: list[GameMap2SVM],
        validation_config: SVMValidation,
        progress_bar_colour: str = "#ed95ce",
    ):
        """
        Evaluate model using symbolic execution engine. It runs in parallel.

        Parameters
        ----------
        maps : list[GameMap2SVM]
            List of maps description.
        validation_svm : SVMValidation
            Validation info.
        progress_bar_colour : str
            Your favorite colour for progress bar.
        """
        with mp.Manager() as sync_manager:
            self._game_manager = self._get_game_manager(validation_config, sync_manager)
            with mp.Pool(validation_config.process_count) as p:
                all_results: list[Map2Result | Exception] = list()
                for result in tqdm.tqdm(
                    p.imap_unordered(self._evaluate_game_map, maps, chunksize=1),
                    desc="validation",
                    total=len(maps),
                    ncols=100,
                    colour=progress_bar_colour,
                ):
                    all_results.append(result)
        return all_results

    def _get_game_manager(
        self, validation_config: SVMValidation, sync_manager: SyncManager
    ) -> BaseGameManager:
        # TODO: docs

        namespace = sync_manager.Namespace()
        namespace.shared_lock = sync_manager.Lock()
        namespace.is_prepared = sync_manager.Value("b", False)
        if isinstance(validation_config, SVMValidationSendEachStep):
            return EachStepGameManager(TrainingModelWrapper(self.model), namespace)
        elif isinstance(validation_config, SVMValidationSendModel):
            return ModelGameManager(namespace, self.model)
        raise RuntimeError(f"There is no game manager suitable to {validation_config}")
