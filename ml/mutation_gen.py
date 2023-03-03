from collections import defaultdict
from dataclasses import dataclass
from typing import Type, TypeAlias

from common.game import GameMap, MoveReward
from ml.model_wrappers.protocols import Mutable

CoveragePercent: TypeAlias = float
StepsCount: TypeAlias = int


@dataclass
class MutableResult:
    move_reward: MoveReward
    steps_count: int
    coverage_percent: float


@dataclass
class MutableResultMapping:
    mutable: Mutable
    mutable_result: MutableResult


GameMapsModelResults: TypeAlias = defaultdict[GameMap, list[MutableResultMapping]]


@dataclass
class MutationProportions:
    n_tops: int
    averaged_n_tops: int
    n_averaged_all: int
    random_n_tops_averaged_mutations: int
    random_all_averaged_mutations: int


@dataclass
class MutatorConfig:
    proportions: MutationProportions
    mutation_volume: float  # 0-1
    mutation_freq: float  # 0-1

    def __post_init__(self):
        in_percents = lambda x: x >= 0 and x <= 1
        if not in_percents(self.mutation_volume):
            raise ValueError(
                f"mutation volume is not in percents: {self.mutation_volume=}"
            )
        if not in_percents(self.mutation_freq):
            raise ValueError(f"mutation freq is not in percents: {self.mutation_freq=}")


class Mutator:
    def __init__(self, config: MutatorConfig, mutable_type: Type[Mutable]) -> None:
        self.config = config
        self.mutable_type = mutable_type

    def n_tops(
        self, game_map_model_results: GameMapsModelResults, n: int
    ) -> list[Mutable]:
        # топ по каждой карте
        all_model_results: list[Mutable] = []

        for model_result_list in game_map_model_results.values():
            all_model_results.extend(model_result_list)

        def model_result_by_reward_asc_steps_desc(
            mutable_mapping: MutableResultMapping,
        ):
            # sort by <MoveRewardReward, -StepsCount (less is better)>
            result = mutable_mapping.mutable_result
            return (
                result.move_reward.ForCoverage,
                result.move_reward.ForVisitedInstructions,
                -result.steps_count,
            )

        n_tops = [
            mapping.mutable
            for mapping in sorted(
                all_model_results,
                key=lambda mr: model_result_by_reward_asc_steps_desc(mr),
            )[:n]
        ]

        return n_tops

    def averaged_n_tops(
        self, game_map_model_results: GameMapsModelResults, n: int
    ) -> Mutable:
        # среднее по топам
        return self.mutable_type.average_n_mutables(
            self.n_tops(game_map_model_results, n)
        )

    def averaged_all(self, game_map_model_results: GameMapsModelResults) -> Mutable:
        # среднее по всем отобранным нейронкам

        all: list[Mutable] = []
        for model_results_mapping_list_for_map in game_map_model_results.values():
            all.extend(map(lambda t: t.mutable, model_results_mapping_list_for_map))

        return self.mutable_type.average_n_mutables(all)

    def random_n_tops_averaged_mutations(
        self, game_map_model_results: GameMapsModelResults, n: int
    ) -> list[Mutable]:
        # случайные мутации среднего по топам
        return self.mutable_type.mutate(
            mutable=self.averaged_n_tops(game_map_model_results, n),
            mutation_volume=self.config.mutation_volume,
            mutation_freq=self.config.mutation_freq,
        )

    def random_all_averaged_mutations(
        self, game_map_model_results: GameMapsModelResults
    ) -> Mutable:
        # случайные мутации среднего по всем отобранным нейронкам
        return self.mutable_type.mutate(
            mutable=self.averaged_all(game_map_model_results),
            mutation_volume=self.config.mutation_volume,
            mutation_freq=self.config.mutation_freq,
        )

    def new_generation(
        self,
        game_map_model_results: GameMapsModelResults,
    ) -> list[Mutable]:
        new_gen = (
            self.n_tops(game_map_model_results, self.config.proportions.n_tops)
            + [
                self.averaged_n_tops(
                    game_map_model_results, self.config.proportions.n_tops
                )
                for _ in range(self.config.proportions.averaged_n_tops)
            ]
            + [
                self.averaged_all(game_map_model_results)
                for _ in range(self.config.proportions.n_averaged_all)
            ]
            + [
                self.random_n_tops_averaged_mutations(
                    game_map_model_results, self.config.proportions.n_tops
                )
                for _ in range(self.config.proportions.random_n_tops_averaged_mutations)
            ]
            + [
                self.random_all_averaged_mutations(game_map_model_results)
                for _ in range(self.config.proportions.random_all_averaged_mutations)
            ]
        )

        return new_gen
