from attrs import define


class StrategyOptions:
    def __init__(self, model_path: str = None):
        self.model_path = model_path

    def __str__(self):
        return f"model_path={self.model_path}"


@define
class BasePSStrategy:
    """Path selection strategy"""

    name: str

    @staticmethod
    def parse(name: str, strategy_options: StrategyOptions) -> "BasePSStrategy":
        """
        Parse strategy from name and kwargs
        :param name: strategy name (AI, ETCC)
        :param kwargs: strategy arguments
        """
        if name == "AI":
            assert strategy_options.model_path is not None
            return AIStrategy(name, strategy_options.model_path)

        if name == "ETCC":
            return ExecutionTreeContributedCoverageStrategy(name)

        raise ValueError(f"Unknown strategy: {name}")


@define
class AIStrategy(BasePSStrategy):
    model_path: str


@define
class ExecutionTreeContributedCoverageStrategy(BasePSStrategy):
    pass
