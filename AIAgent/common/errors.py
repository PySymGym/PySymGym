from common.game import GameMap


class GamesError(ExceptionGroup):
    def __new__(cls, errors: list[Exception], maps: list[GameMap]):
        self = super().__new__(GamesError, "There are failed or timeouted maps", errors)
        self.maps = maps
        return self

    def derive(self, exc):
        return GamesError(self.message, exc)
