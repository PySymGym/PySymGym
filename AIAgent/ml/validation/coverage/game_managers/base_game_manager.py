from abc import ABC, abstractmethod
from multiprocessing.managers import Namespace
from typing import Optional

from common.classes import Map2Result
from common.game import GameMap, GameMap2SVM
from torch_geometric.data.hetero_data import HeteroData


class BaseGamePreparator(ABC):
    """Abstract class for preparing the game environment before starting the game.

    This class provides a basic structure for performing preparatory actions
    required before starting the game process. The preparation is executed
    in a thread-safe manner, ensuring that the process is performed only once.
    """

    def __init__(self, namespace: Namespace):
        """
        Initializes the base game preparator.

        Attributes:
            _is_prepared: Flag indicating whether the preparation is complete.
            _shared_lock: Thread lock to ensure thread-safe preparation.
        """
        self._shared_lock = namespace.shared_lock
        self._is_prepared = namespace.is_prepared

    def prepare(self):
        """
        Ensures that the preparation process is executed only once in a thread-safe manner.

        This method allows only one thread to perform the preparation and blocks
        all attempts after the first thread completes it.
        """
        with self._shared_lock:
            if self._is_prepared.value:
                return
            self._prepare()
            self._is_prepared.value = True
            return

    @abstractmethod
    def _prepare(self):
        """
        Prepares the game environment before starting the game.
        """
        ...


class BaseGameManager(ABC):
    def __init__(self, namespace: Namespace):
        self._namespace = namespace
        self._preparator = self._create_preparator()

    def play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        """Executes GameMap on symbolic engine"""
        self._preparator.prepare()
        return self._play_game_map(game_map2svm)

    @abstractmethod
    def _create_preparator(self) -> BaseGamePreparator: ...

    @abstractmethod
    def _play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        """Executes GameMap on symbolic engine.

        This method is called by 'play_game_map' after the assurance that preparation step is not skipped.
        """
        ...

    @abstractmethod
    def get_game_steps(self, game_map: GameMap) -> Optional[list[HeteroData]]:
        """Returns list of HeteroData for each step. Returns None if steps of game_map can't be received."""
        ...

    @abstractmethod
    def delete_game_artifacts(self, game_map: GameMap):
        """Deletes game artifacts of game_map"""
        ...
