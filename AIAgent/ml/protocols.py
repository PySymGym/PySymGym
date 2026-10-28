from abc import ABC, abstractmethod
from typing import Protocol

import torch

from common.game import GameState


class Named(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class Predictor(Named, ABC):
    @abstractmethod
    def predict(self, input: GameState, map_name):
        raise NotImplementedError

    @abstractmethod
    def model(self) -> torch.nn.Module:
        raise NotImplementedError


class Model(Protocol):
    """The forward contract every registered model must satisfy.

    ``ml.inference.infer`` is the sole caller: it invokes ``forward`` with
    exactly these keyword arguments and expects a ``(num_states, 1)``
    tensor back. The registry tests pin each registered model's ``forward``
    signature against this one, so a non-conformant model fails at test
    time instead of on the first training batch.
    """

    def forward(
        self,
        game_x: torch.Tensor,
        state_x: torch.Tensor,
        pc_x: torch.Tensor,
        edge_index_v_v: torch.Tensor,
        edge_type_v_v: torch.Tensor,
        edge_index_history_v_s: torch.Tensor,
        edge_attr_history_v_s: torch.Tensor,
        edge_index_in_v_s: torch.Tensor,
        edge_index_s_s: torch.Tensor,
        edge_index_pc_pc: torch.Tensor,
        edge_index_pc_s: torch.Tensor,
    ) -> torch.Tensor: ...
