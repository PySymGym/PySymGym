import torch


def get_device() -> torch.device:
    """Return the torch device to run on, resolved at call time.

    Importing this module must never probe CUDA. The device is computed on
    every call so tests can patch ``torch.cuda.is_available`` and so importing
    AIAgent code works on machines without a GPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
