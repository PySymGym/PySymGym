"""Tests for the lazy device seam in ``ml.device`` (``get_device`` and
``cuda_is_usable``) and the timestamped state-dict save in
``ml.model_io.save_state_dict``.

The device must be resolved when it is used, not when ``ml.device`` is
imported, so importing AIAgent code never probes CUDA and tests can force
CPU.
"""

import re
from pathlib import Path

import pytest
import torch

from ml.device import cuda_is_usable, get_device
from ml.model_io import save_state_dict

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


def test_cuda_is_usable_false_without_a_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert cuda_is_usable() is False


def test_cuda_is_usable_false_when_the_op_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def no_kernel_image(*args, **kwargs):
        raise RuntimeError(
            "CUDA error: no kernel image is available for execution on the device"
        )

    monkeypatch.setattr(torch, "randn", no_kernel_image)

    assert cuda_is_usable() is False


def test_cuda_is_usable_true_when_the_op_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # A CPU stand-in: the op path (matmul, sum, sync) still runs end to end.
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.zeros(64, 64))

    assert cuda_is_usable() is True


def test_save_state_dict_writes_timestamped_state_dict(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)

    save_state_dict(model, tmp_path / "saved", with_name="agent")

    files = list((tmp_path / "saved").glob("agent_*.pt"))
    assert len(files) == 1
    # str(datetime) format: YYYY-MM-DD HH:MM:SS.ffffff (wall-clock, so the
    # pattern is asserted, not the exact timestamp).
    assert re.fullmatch(
        r"agent_\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\.pt", files[0].name
    )
    loaded = torch.load(files[0], weights_only=True)
    assert set(loaded) == set(model.state_dict())
    for key, tensor in model.state_dict().items():
        assert torch.equal(loaded[key], tensor)
