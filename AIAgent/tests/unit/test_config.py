"""Tests for the lazy device seam in ``config.get_device``.

The device must be resolved when it is used, not when ``config`` is imported,
so importing AIAgent code never probes CUDA and tests can force CPU.
"""

import pytest
import torch

from config import GeneralConfig, get_device

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, "cuda:0"), (False, "cpu")],
)
def test_get_device_selects_available_backend(
    cuda_available: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert get_device() == torch.device(expected)


def test_device_is_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cuda:0")


def test_no_import_time_device_binding() -> None:
    assert not hasattr(GeneralConfig, "DEVICE")
