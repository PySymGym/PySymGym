from func_timeout import FunctionTimedOut
from common.classes import Map2Result
from connection.errors_connection import GameInterruptedError
from common.utils import inheritors
from common.game import GameMap2SVM


class GameError(Exception):

    def __init__(
        self,
        map2result: Map2Result,
        error_name: str,
    ) -> None:
        self.map2result = map2result
        self._error_name = error_name

        super().__init__(map2result, error_name)

    def need_to_save_map(self):
        gie_inheritors = inheritors(GameInterruptedError)
        need_to_save_classes = list(gie_inheritors) + [FunctionTimedOut]
        return self._error_name in map(lambda it: it.__name__, need_to_save_classes)
