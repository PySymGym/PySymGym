from pathlib import Path
from attrs import define


@define
class BasePSStrategy:
    """Path selection strategy"""

    name: str

    @staticmethod
    def parse(name: str, model_path: Path = None) -> "BasePSStrategy":
        """
        Parse strategy from name and kwargs
        :param name: strategy name (AI, ETCC)
        :param kwargs: strategy arguments
        """
        if name == "AI":
            assert model_path is not None
            return AIStrategy(name, model_path)

        if name == "ExecutionTreeContributedCoverage":
            return ExecutionTreeContributedCoverageStrategy(name)

        raise ValueError(f"Unknown strategy: {name}")


@define
class AIStrategy(BasePSStrategy):
    model_path: str


@define
class ExecutionTreeContributedCoverageStrategy(BasePSStrategy):
    pass
