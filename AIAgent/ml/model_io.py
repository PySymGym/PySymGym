from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage


def save_state_dict(model: torch.nn.Module, save_path: Path, with_name: str) -> None:
    """Save ``model``'s state dict to ``save_path`` under a timestamped name.

    The file is ``{with_name}_{timestamp}.pt`` where the timestamp is the
    wall-clock ``str(datetime)`` (``YYYY-MM-DD HH:MM:SS.ffffff``).
    """
    save_path.mkdir(exist_ok=True)
    timestamp = datetime.fromtimestamp(datetime.now().timestamp())

    torch.save(model.state_dict(), save_path / f"{with_name}_{timestamp}.pt")


#: The torch-geometric storage classes a pickled ``HeteroData`` (and a
#: checkpoint built from one) may contain. A bare ``torch.load`` under the
#: secure unpickler rejects them; every safe load in the project allows
#: exactly these (torch's own classes are always allowed).
STORAGE_SAFE_GLOBALS = (BaseStorage, NodeStorage, EdgeStorage)


def safe_torch_load(path: str | Path, map_location: torch.device | str) -> Any:
    """``torch.load`` with the project's storage-class allow-list.

    The ``safe_globals`` context allows exactly ``STORAGE_SAFE_GLOBALS`` for
    the duration of the load, so pickled torch-geometric objects deserialize
    under the secure unpickler. ``map_location`` remaps the loaded tensors to
    the caller's device: a checkpoint saved on another device (a CUDA-saved
    model on a CPU-only machine) crashes a bare load (#667).

    Parameters
    ----------
    path:
        The checkpoint file to load.
    map_location:
        The device the loaded tensors are mapped to (typically
        ``ml.device.get_device()``).

    Returns
    -------
    The object stored at ``path``.
    """
    # torch's parameter is an invariant list[Callable | tuple[Callable, str]];
    # the tuple-to-list conversion at the call site satisfies the annotation
    # without changing what the unpickler allows.
    with torch.serialization.safe_globals(list(STORAGE_SAFE_GLOBALS)):
        return torch.load(path, map_location=map_location)
