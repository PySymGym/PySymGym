from abc import ABC, abstractmethod

from common.classes import Map2Result
from common.game import GameMap, GameMap2SVM
from torch_geometric.data.hetero_data import HeteroData


class BaseGameManager(ABC):
    @abstractmethod
    def play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        """Executes GameMap on symbolic engine

        Args:
            game_map2svm (GameMap2SVM)

        Returns:
            Map2Result
        """
        ...

    @abstractmethod
    def get_game_states(self, game_map: GameMap) -> list[HeteroData]:
        """`get_game_status` guarantees to receive the steps of the played game no more than once

        Args:
            game_map (`GameMap`)

        Returns:
            List with steps (`list[HeteroData]`)
        """
        ...
