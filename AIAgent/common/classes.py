from collections import defaultdict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, TypeAlias, Union

from common.game import GameMap, GameMap2SVM
from ml.protocols import Named


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

    def printable(self, verbose) -> str:
        steps_format = (
            f"steps: {self.steps_count}," if verbose else f"#s={self.steps_count}"
        )
        tests_count_format = (
            f"test count: {self.tests_count}" if verbose else f"#t={self.tests_count}"
        )
        errors_count_format = (
            f"error count: {self.errors_count}"
            if verbose
            else f"#e={self.errors_count}"
        )
        actual_coverage_percent_format = (
            f"actual %: {self.actual_coverage_percent:.2f},"
            if verbose
            else f"%ac={self.actual_coverage_percent:.2f}"
        )
        return f"{actual_coverage_percent_format} {steps_format} {tests_count_format} {errors_count_format}"


class GameFailed:
    def __str__(self) -> str:
        return "FAILED"


@dataclass
class Agent2Result:
    agent: Named
    game_result: GameResult


@dataclass_json
@dataclass
class Map2Result:
    map: GameMap2SVM
    game_result: Union[GameResult, GameFailed]


GameMapsModelResults: TypeAlias = defaultdict[GameMap, list[Agent2Result]]
AgentResultsOnGameMaps: TypeAlias = defaultdict[Named, list[Map2Result]]
