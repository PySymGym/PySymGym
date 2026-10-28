from datetime import datetime
from pathlib import Path

import torch


def save_model(model: torch.nn.Module, /, **initargs):
    weights = model.state_dict()

    # ml.models.NorthernPenguin.model
    save_path_components = model.__module__.split(".")[:-1]

    # ml.models.NorthernPenguin.model.StateModelEncoder
    class_fullname = model.__module__ + "." + model.__class__.__name__

    # **{hidden_channels: 32, out_channels: 8}
    model_initargs = "_".join([f"{param}_{value}" for param, value in initargs.items()])

    save_dir = Path("/".join(save_path_components))

    timestamp = datetime.fromtimestamp(datetime.now().timestamp())

    suffix = ".pt"

    save_name = f"{class_fullname}{'_' + model_initargs + '_' if initargs else ''}{timestamp}{suffix}"

    torch.save(weights, save_dir / save_name)
