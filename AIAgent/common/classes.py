from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeAlias

from dataclasses_json import dataclass_json
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from connection.broker_conn.classes import SVMInfo
from common.game import GameMap
from ml.protocols import Named

PlatformName: TypeAlias = str
SVMInfoName: TypeAlias = str


@dataclass_json
@dataclass
class GameResult:
    steps_count: int
    tests_count: int
    errors_count: int
    actual_coverage_percent: Optional[float] = None

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


@dataclass
class Agent2Result:
    agent: Named
    game_result: GameResult


@dataclass_json
@dataclass
class Map2Result:
    map: GameMap
    game_result: GameResult


GameMapsModelResults: TypeAlias = defaultdict[GameMap, list[Agent2Result]]
AgentResultsOnGameMaps: TypeAlias = defaultdict[Named, list[Map2Result]]


@pydantic_dataclass
class SVMConfig:
    platform_name: PlatformName
    epochs: int
    SVMInfo: SVMInfo


@pydantic_dataclass
class DatasetConfig:
    dataset_base_path: Path  # path to dir with explored dlls
    dataset_description: Path  # full paths to JSON-file with dataset description

    @field_validator("dataset_base_path", "dataset_description", mode="before")
    @classmethod
    def transform(cls, input: str) -> Path:
        return Path(input).resolve()


@pydantic_dataclass
class Platform:
    name: PlatformName
    DatasetConfig: DatasetConfig


@pydantic_dataclass
class OptunaConfig:
    n_startup_trials: int  # number of optuna's trials
    n_trials: int  # number of optuna's trials


@pydantic_dataclass
class Config:
    SVMConfigs: list[SVMConfig]
    Platforms: list[Platform]
    OptunaConfig: OptunaConfig
    path_to_weights: Optional[Path] = Field(default=None)

    @field_validator("path_to_weights", mode="before")
    @classmethod
    def transform(cls, input: Optional[str]) -> Optional[Path]:
        return Path(input).absolute() if input is not None else None
