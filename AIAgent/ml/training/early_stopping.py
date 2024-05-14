class EarlyStopping:
    def __init__(self, state_len: int = 5, tolerance: float = 0.01) -> None:
        self._state_len = state_len
        self._state = list()
        self._tolerance = tolerance

    def is_continue(self, state_value) -> bool:
        if len(self._state) < self._state_len:
            self._state.append(state_value)
            return True

        if abs((sum(self._state) / self._state_len) - state_value) < self._tolerance:
            return False
        else:
            return True
