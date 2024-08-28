from ml.training.epochs_statistics import StatisticsCollector
from connection.errors_connection import GameInterruptedError
from common.utils import inheritors
from common.game import GameMap2SVM


class GameError(Exception):

    def __init__(
        self,
        game_map2svm: GameMap2SVM,
        error_name: str,
    ) -> None:
        self._map = game_map2svm
        self._error_name = error_name

        super().__init__(game_map2svm, error_name)

    def handle_error(self, statistics_collector: StatisticsCollector):
        gie_inheritors = inheritors(GameInterruptedError)
        if self._error_name in map(lambda it: it.__name__, gie_inheritors):
            return
        statistics_collector.fail(self._map)
