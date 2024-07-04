from common.game import GameMap


class GameError(RuntimeError):
    def __init__(self, message, maps: list[GameMap]):
        super().__init__(message)
        self.maps = maps
