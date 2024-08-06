from common.game import GameMap2SVM


class GameError(Exception):
    def __init__(self, game_map2svm: GameMap2SVM, error: Exception) -> None:
        self._map = game_map2svm
        self._error = error
        super().__init__(GameError, self._error)
