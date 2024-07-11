from common.game import GameMap


class GameErrors(ExceptionGroup):
    def __new__(cls, errors: list[Exception], maps: list[GameMap]):
        self = super().__new__(GameErrors, "There are failed or timeouted maps", errors)
        self.maps = maps
        return self

    def derive(self, excs):
        return GameErrors(self.message, excs)
