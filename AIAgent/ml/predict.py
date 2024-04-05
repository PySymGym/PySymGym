from collections import namedtuple

import torch
from torch_geometric.data import HeteroData

from ml.inference import infer
from config import GeneralConfig

StateVectorMapping = namedtuple("StateVectorMapping", ["state", "vector"])


def predict_state_with_dict(
    model: torch.nn.Module, data: HeteroData, state_map: dict[int, int]
) -> int:
    """Gets state id from model and heterogeneous graph
    data.state_map - maps real state id to state index"""

    data.to(GeneralConfig.DEVICE)
    reversed_state_map = {v: k for k, v in state_map.items()}

    with torch.no_grad():
        out = infer(model, data)

    remapped = []

    for index, vector in enumerate(out):
        state_vector_mapping = StateVectorMapping(
            state=reversed_state_map[index],
            vector=(vector.detach().cpu().numpy()).tolist(),
        )
        remapped.append(state_vector_mapping)

    return max(remapped, key=lambda mapping: sum(mapping.vector)).state, out
