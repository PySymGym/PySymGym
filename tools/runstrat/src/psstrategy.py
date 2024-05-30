import attrs


@attrs.define
class BasePSStrategy:
    """Path selection strategy"""

    name: str

    @staticmethod
    def parse(name: str, **kwargs):
        assert len(kwargs) <= 1, f"too many arguments in {kwargs=}"
        if name == "AI":
            assert kwargs["model_path"], f"No model_path is provided in {kwargs=}"
            return AIStrategy(name, kwargs["model_path"])

        return BasePSStrategy(name=name)


@attrs.define
class AIStrategy(BasePSStrategy):
    model_path: str
