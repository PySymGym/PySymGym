"""Probe whether the locked torch can use the visible GPU (CI, #662).

The e2e container may have a GPU device passed through (the ``detect-gpu``
job) that the locked torch build cannot run on: a compute capability the
wheel has no kernels for makes the first CUDA op crash mid-training with
"no kernel image is available for execution on the device" instead of
degrading to CPU. This probe runs in the pipeline after the Python
environment is prepared, so it checks the exact torch the pipeline trains
with: it runs a small CUDA op on ``cuda:0`` and, when the op fails, hides
the device from torch for the rest of the run (an empty
``CUDA_VISIBLE_DEVICES`` via ``$GITHUB_ENV``) so ``ml.device.get_device``
falls back to CPU.

The verdict is the op, not the arch-list membership: a compute capability
absent from ``torch.cuda.get_arch_list()`` can still work through driver
JIT of bundled PTX (the locked torch 2.7.1 runs on sm_61 this way), and an
arch-list hit proves nothing if the op fails. The arch list is logged for
diagnosis. The probe never fails the run — every path exits 0; the loud
WARNING is the signal (the detect-gpu "degrade instead of failing"
philosophy).

Usage (from the repository root, inside the e2e container):
    poetry run python AIAgent/probe_gpu_torch.py
"""

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GpuInfo:
    """The probe facts: the locked torch build and the visible GPU."""

    torch_version: str
    arch_list: tuple[str, ...]
    device_name: str
    capability: tuple[int, int]
    op_error: str | None  # None when the small CUDA op succeeded


def evaluate(info: GpuInfo) -> bool:
    """Return True when the pipeline may use the GPU.

    The verdict is the small CUDA op (see the module docstring for why the
    arch-list membership is not the verdict).
    """
    return info.op_error is None


def describe(info: GpuInfo) -> str:
    """One-line summary of the probe facts, for the CI log."""
    return (
        f"{info.device_name} (compute capability sm_{info.capability[0]}{info.capability[1]}) "
        f"vs torch {info.torch_version} (arch list: {', '.join(info.arch_list)})"
    )


def _run_op() -> None:
    """Run a small CUDA op on ``cuda:0``; raise when the GPU cannot run it."""
    x = torch.randn(64, 64, device="cuda:0")
    (x @ x).sum().item()


def _hide_device_from_torch() -> None:
    """Hide the GPU from torch for the rest of the CI run.

    Appends an empty ``CUDA_VISIBLE_DEVICES`` to ``$GITHUB_ENV`` so every
    later step sees no CUDA device. Outside GitHub Actions (no
    ``GITHUB_ENV``) there is nothing to do — the probe is a CI diagnostic.
    """
    github_env = os.environ.get("GITHUB_ENV")
    if github_env is None:
        return
    with open(github_env, "a", encoding="utf-8") as handle:
        handle.write("CUDA_VISIBLE_DEVICES=\n")


def main() -> int:
    if not torch.cuda.is_available():
        print("probe_gpu_torch: no CUDA device visible; the pipeline runs the CPU flow")
        return 0

    op_error = None
    try:
        _run_op()
    except Exception as exc:
        # Any CUDA failure (missing kernel image, driver error, ...) means
        # this GPU cannot be trusted for the run: degrade, never fail.
        op_error = f"{type(exc).__name__}: {exc}"

    info = GpuInfo(
        torch_version=torch.__version__,
        arch_list=tuple(torch.cuda.get_arch_list()),
        device_name=torch.cuda.get_device_name(0),
        capability=tuple(torch.cuda.get_device_capability(0)),
        op_error=op_error,
    )
    if evaluate(info):
        print(
            f"probe_gpu_torch: GPU compatible with the locked torch: {describe(info)}"
        )
        return 0

    print(
        f"WARNING: the visible GPU is incompatible with the locked torch: {describe(info)}"
    )
    print(f"probe_gpu_torch: the CUDA op failed with: {info.op_error}")
    print(
        "probe_gpu_torch: degrading to the CPU flow; hiding the device from torch "
        "for the rest of the run"
    )
    _hide_device_from_torch()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
