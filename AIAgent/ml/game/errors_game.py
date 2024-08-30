from func_timeout import FunctionTimedOut
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

    def need_to_save_map(self):
        gie_inheritors = inheritors(GameInterruptedError)
        need_to_save_classes = list(gie_inheritors) + [FunctionTimedOut]
        return self._error_name in map(lambda it: it.__name__, need_to_save_classes)
