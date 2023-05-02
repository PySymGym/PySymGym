import random
import string
from common.constants import Constant
from copy import deepcopy
from math import floor

import numpy as np
import numpy.typing as npt

from common.game import GameState
from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.predict_state_vector_hetero import PredictStateVectorHetGNN
from ml.utils import load_full_model

from .protocols import ModelWrapper, Mutable

MAX_W, MIN_W = 1, -1


class GeneticLearner(ModelWrapper):
    MODEL = None

    def name(self) -> str:
        return self._name

    @staticmethod
    def set_static_model():
        GeneticLearner.MODEL = load_full_model(Constant.IMPORTED_FULL_MODEL_PATH)

    def __init__(self, weights: npt.NDArray = None) -> None:
        if weights is None:
            # -1 to 1
            self.weights = np.random.rand(Constant.NUM_FEATURES) * 2 - 1
        else:
            self.weights = weights

        self._name = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=7)
        )

    def __str__(self) -> str:
        return f"{self.name()}: {self.weights.tolist()}"

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def predict(self, input: GameState):
        hetero_input, state_map = ServerDataloaderHeteroVector.convert_input_to_tensor(
            input
        )
        assert GeneticLearner.MODEL is not None
        next_step_id = PredictStateVectorHetGNN.predict_state_weighted(
            GeneticLearner.MODEL, self.weights, hetero_input, state_map
        )
        return next_step_id

    @staticmethod
    def average(ms: list[Mutable]) -> Mutable:
        mutables_weights = [model.weights for model in ms]
        return GeneticLearner(
            weights=np.mean(mutables_weights, axis=0),
        )

    @staticmethod
    def mutate(
        mutable: Mutable, mutation_volume: float, mutation_freq: float
    ) -> Mutable:
        """
        mutation_volume - 0..1, percentage of components of the weights vector to mutate
        mutation_freq - 0..1, variation of weights, within (MAX_W, MIN_W)
        """
        assert mutation_freq < MAX_W and mutation_freq > MIN_W
        new_mutable = deepcopy(mutable)
        to_mutate = floor(Constant.NUM_FEATURES * mutation_volume)

        for _ in range(to_mutate):
            index_to_mutate = random.randint(0, Constant.NUM_FEATURES - 1)
            new_mutable.weights[index_to_mutate] = variate(
                val=mutable.weights[index_to_mutate],
                range_percent=mutation_freq,
                borders=(MIN_W, MAX_W),
            )

        return new_mutable

    def train_single_val(self):
        return super().train_single_val()


def variate(val: float, range_percent: float, borders: tuple[float, float]):
    sign = 1 if random.random() - 0.5 > 0 else -1
    border_range = borders[1] - borders[0]
    variated = val + sign * range_percent * border_range
    if variated > borders[1]:
        return borders[1]
    if variated < borders[0]:
        return borders[0]
    return variated
