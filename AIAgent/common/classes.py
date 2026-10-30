from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, Union

from common.game import GameMap2SVM


@dataclass_json
@dataclass
class GameResult:
    steps_count: int
    tests_count: int
    errors_count: int
    actual_coverage_percent: Optional[float] = None

    def __str__(self) -> str:
        return str(
            (
                self.actual_coverage_percent,
                self.tests_count,
                self.steps_count,
                self.errors_count,
            )
        )


class GameFailed:
    def __init__(self, reason) -> None:
        self.reason = reason

    def __str__(self) -> str:
        return "FAILED"


@dataclass_json
@dataclass
class Map2Result:
    map: GameMap2SVM
    game_result: Union[GameResult, GameFailed]
