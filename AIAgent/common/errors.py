from common.game import GameMap, GameMap2SVM


class GameErrors(ExceptionGroup):
    def __new__(cls, errors: list[Exception], maps: list[GameMap]):
        self = super().__new__(GameErrors, "There are failed or timeouted maps", errors)
        self.maps = maps
        return self

    def derive(self, excs):
        return GameErrors(self.message, excs)


class GameError(Exception):
    def __init__(self, game_map2svm: GameMap2SVM, error: Exception) -> None:
        self._map = game_map2svm
        self._error = error
        super().__init__(self._error.args)
