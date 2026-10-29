from datetime import datetime
from pathlib import Path

import torch


def save_state_dict(model: torch.nn.Module, save_path: Path, with_name: str) -> None:
    """Save ``model``'s state dict to ``save_path`` under a timestamped name.

    The file is ``{with_name}_{timestamp}.pt`` where the timestamp is the
    wall-clock ``str(datetime)`` (``YYYY-MM-DD HH:MM:SS.ffffff``).
    """
    save_path.mkdir(exist_ok=True)
    timestamp = datetime.fromtimestamp(datetime.now().timestamp())

    torch.save(model.state_dict(), save_path / f"{with_name}_{timestamp}.pt")
