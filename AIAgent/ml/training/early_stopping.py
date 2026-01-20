from common.config.optuna_config import OptimizationDirection


class EarlyStopping:
    def __init__(
        self,
        state_len: int = 5,
        tolerance: float = 0.01,
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
    ) -> None:
        self._state_len = state_len
        self._state = list()
        self._tolerance = tolerance
        match direction:
            case OptimizationDirection.MINIMIZE:
                self._get_difference = (
                    lambda state_value: (sum(self._state) / self._state_len)
                    - state_value
                )
            case OptimizationDirection.MAXIMIZE:
                self._get_difference = lambda state_value: state_value - (
                    sum(self._state) / self._state_len
                )

    def is_continue(self, state_value) -> bool:
        if len(self._state) < self._state_len:
            self._state.append(state_value)
            return True

        if self._get_difference(state_value) < self._tolerance:
            return False
        else:
            self._state.pop(0)
            self._state.append(state_value)
            return True
