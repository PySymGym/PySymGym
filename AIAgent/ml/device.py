import torch


def get_device() -> torch.device:
    """Return the torch device to run on, resolved at call time.

    Importing this module must never probe CUDA. The device is computed on
    every call so tests can patch ``torch.cuda.is_available`` and so importing
    AIAgent code works on machines without a GPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cuda_is_usable() -> bool:
    """Whether a CUDA op can actually run on the visible device.

    ``torch.cuda.is_available()`` reports a driver-visible device, not a
    usable one: when the locked torch has no kernels for the GPU's compute
    capability, the device is visible but every CUDA op fails with "no
    kernel image is available for execution on the device". Like
    ``probe_gpu_torch.py`` (#662), the verdict is a small real op on
    ``cuda:0``, not the visibility flag.
    """
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(64, 64, device="cuda:0")
        (x @ x).sum().item()
    except Exception:
        # Any CUDA failure (missing kernel image, driver error, ...) means
        # the visible device cannot be trusted: report it as unusable.
        return False
    return True
