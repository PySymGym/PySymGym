"""Select the torch device the installed build can actually run on.

``torch.cuda.is_available()`` returns True as soon as a CUDA device is
visible, even when the installed torch build ships no kernels for the
device's compute capability: torch >=2.9 dropped Pascal (sm_61), so on a
Pascal GPU a newer torch selects ``cuda:0`` and dies on the first batch
with "no kernel image is available for execution on the device" (issue
#644, observed in the PR #610 CI runs). This module is the single probe
for the real question — can the installed build run a kernel on the
visible GPU? — by launching a tiny smoke kernel and degrading to CPU
when the build has no kernels for the device.

Usage (e2e workflow step "Check GPU compatibility"):
    python3 device.py

The script always exits 0: the probe is informational, and the CPU
fallback is applied automatically by ``config.py``.
"""

import torch


def _smoke_kernel() -> None:
    """Run a tiny matmul on ``cuda:0`` and synchronize.

    The ``.item()`` call forces a device sync, so a kernel-launch failure
    (a CUDA error — ``RuntimeError`` or ``torch.AcceleratorError``
    depending on the torch version — when the build has no kernels for the
    device) surfaces here instead of on the first real training batch.
    """
    tensor = torch.ones(8, 8, device="cuda:0")
    torch.matmul(tensor, tensor).sum().item()


def _gpu_facts() -> str:
    """Return ``"<name> (sm_<major><minor>)"`` for ``cuda:0``.

    Falls back to ``"unknown GPU"`` when the driver queries fail, so a
    broken context can never mask the CPU fallback.
    """
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        return f"{name} (sm_{major}{minor})"
    except Exception:
        return "unknown GPU"


def _supported_archs() -> str:
    """Return the space-joined arch list of the installed torch build."""
    try:
        return " ".join(torch.cuda.get_arch_list())
    except Exception:
        return "unknown"


def _probe() -> tuple[torch.device, str]:
    """Probe the installed build and return ``(device, verdict_line)``.

    The verdict line is a single stdout line, stable for log grepping:
    ``device=cuda:0 (...)`` when the build runs on the GPU,
    ``device=cpu (no CUDA device)`` when no CUDA device is visible, and
    ``device=cpu (GPU ... not supported by torch ...; ...)`` when the
    build has no kernels for the visible GPU.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), "device=cpu (no CUDA device)"
    try:
        _smoke_kernel()
    except Exception as error:
        first_line = " ".join(str(error).split())
        line = (
            f"device=cpu (GPU {_gpu_facts()} not supported by torch "
            f"{torch.__version__} [supports: {_supported_archs()}]; {first_line})"
        )
        return torch.device("cpu"), line
    return torch.device(
        "cuda:0"
    ), f"device=cuda:0 ({_gpu_facts()}, torch {torch.__version__})"


def select_device() -> torch.device:
    """Return the device the installed torch build can actually run on.

    ``cuda:0`` only when a CUDA device is visible *and* a real kernel
    runs on it; ``cpu`` otherwise. When the build has no kernels for the
    visible GPU, the one-line diagnosis is printed so the CPU fallback is
    explained in the log of the process that fell back.

    Returns
    -------
    torch.device
        ``cuda:0`` or ``cpu``.
    """
    cuda_visible = torch.cuda.is_available()
    device, line = _probe()
    if cuda_visible and device.type == "cpu":
        print(line)
    return device


def main() -> None:
    """Print the device verdict line. Always exits 0 (see module docstring)."""
    _, line = _probe()
    print(line)


if __name__ == "__main__":
    main()
