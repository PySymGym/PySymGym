from ml.training.epochs_statistics import StatisticsCollector
from connection.errors_connection import GameInterruptedError
from common.game import GameMap2SVM


class GameError(Exception):
    def __init__(self, game_map2svm: GameMap2SVM, error: Exception) -> None:
        self._map = game_map2svm
        self._error = error
        super().__init__(GameError, self._error)

    def handle_error(self, statistics_collector: StatisticsCollector):
        if isinstance(self._error, GameInterruptedError):
            return
        statistics_collector.fail(self._map)
