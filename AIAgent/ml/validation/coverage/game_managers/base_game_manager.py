from abc import ABC, abstractmethod

from common.classes import Map2Result
from common.game import GameMap, GameMap2SVM
from torch_geometric.data.hetero_data import HeteroData


class BaseGameManager(ABC):
    def __init__(self):
        self._game_states: dict[str, list[HeteroData]] = {}
        self.is_ready_to_play = False

    def play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        """Executes GameMap on symbolic engine"""
        if not self.is_ready_to_play:
            self.prepare()
            self.is_ready_to_play = True
        return self._play_game_map(game_map2svm)

    @abstractmethod
    def _play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        """Executes GameMap on symbolic engine

        This method is called by 'play_game_map' after the assurance that preparation step is not skipped.
        """
        ...

    @abstractmethod
    def prepare(self):
        """Prepare before starting validation"""
        ...

    def get_game_steps(self, game_map: GameMap) -> list[HeteroData]:
        """Returns list of HeteroData for each step. Guarantees to receive the steps of the played game no more than once. Raises a KeyError

        Args:
            game_map (`GameMap`)

        Returns:
            List with steps (`list[HeteroData]`)
        """
        return self._game_states.pop(str(game_map))
