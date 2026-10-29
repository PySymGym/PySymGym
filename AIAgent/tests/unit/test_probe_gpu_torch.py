"""Tests for the CI GPU/torch compatibility probe (``probe_gpu_torch``, #662).

The probe decides whether the e2e pipeline may use the visible GPU with the
locked torch. Every path is drivable without a GPU: the autouse
``cpu_device`` fixture forces ``torch.cuda.is_available`` to False, and the
CUDA-touching seams (the module's ``torch`` global, ``_run_op``) are
replaced with fakes.
"""

import pytest

import probe_gpu_torch
from probe_gpu_torch import (
    GpuInfo,
    _hide_device_from_torch,
    _run_op,
    describe,
    evaluate,
    main,
)

pytestmark = pytest.mark.unit


def _info(op_error: str | None = None, **overrides: object) -> GpuInfo:
    base = dict(
        torch_version="2.7.1+cu126",
        arch_list=("sm_50", "sm_60", "sm_70"),
        device_name="NVIDIA GeForce GTX 1030",
        capability=(6, 1),
        op_error=op_error,
    )
    base.update(overrides)
    return GpuInfo(**base)


class TestEvaluate:
    def test_successful_op_is_compatible(self):
        assert evaluate(_info(op_error=None))

    def test_failed_op_is_incompatible(self):
        assert not evaluate(_info(op_error="CUDAError: no kernel image is available"))

    def test_arch_list_absence_does_not_override_a_successful_op(self):
        # sm_61 is absent from the arch list, yet the op succeeded (driver
        # JIT of bundled PTX): the GPU is usable.
        info = _info(op_error=None, arch_list=("sm_70", "sm_75"))
        assert evaluate(info)


class TestDescribe:
    def test_summary_lists_device_capability_torch_and_arch_list(self):
        assert (
            describe(_info()) == "NVIDIA GeForce GTX 1030 (compute capability sm_61) "
            "vs torch 2.7.1+cu126 (arch list: sm_50, sm_60, sm_70)"
        )


class _FakeTensor:
    def __matmul__(self, other: object) -> "_FakeTensor":
        return self

    def sum(self) -> "_FakeTensor":
        return self

    def item(self) -> float:
        return 0.0


class TestRunOp:
    def test_runs_matmul_on_cuda0(self, monkeypatch: pytest.MonkeyPatch):
        calls: dict[str, object] = {}

        class _Torch:
            @staticmethod
            def randn(*shape: int, **kwargs: object) -> _FakeTensor:
                calls.update(kwargs)
                return _FakeTensor()

        monkeypatch.setattr(probe_gpu_torch, "torch", _Torch)
        _run_op()
        assert calls == {"device": "cuda:0"}


class TestHideDeviceFromTorch:
    def test_appends_empty_cuda_visible_devices(self, tmp_path, monkeypatch):
        env_file = tmp_path / "github_env"
        monkeypatch.setenv("GITHUB_ENV", str(env_file))
        _hide_device_from_torch()
        assert env_file.read_text(encoding="utf-8") == "CUDA_VISIBLE_DEVICES=\n"

    def test_noop_without_github_env(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ENV", raising=False)
        _hide_device_from_torch()  # must not raise


class _FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def get_arch_list(self) -> list[str]:
        return ["sm_70", "sm_75"]

    def get_device_name(self, index: int) -> str:
        return "NVIDIA GeForce GTX 1030"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return (6, 1)


def _fake_torch(available: bool = True):
    class _Torch:
        cuda = _FakeCuda(available)
        __version__ = "2.7.1+cu126"

    return _Torch


class TestMain:
    def test_no_cuda_reports_the_cpu_flow(self, capsys):
        # The autouse cpu_device fixture already forces is_available to False.
        assert main() == 0
        assert "no CUDA device visible" in capsys.readouterr().out

    def test_compatible_gpu_keeps_the_device(self, monkeypatch, capsys, tmp_path):
        env_file = tmp_path / "github_env"
        monkeypatch.setenv("GITHUB_ENV", str(env_file))
        monkeypatch.setattr(probe_gpu_torch, "torch", _fake_torch())
        monkeypatch.setattr(probe_gpu_torch, "_run_op", lambda: None)
        assert main() == 0
        assert "compatible" in capsys.readouterr().out
        assert not env_file.exists()

    def test_incompatible_gpu_degrades_to_cpu(self, monkeypatch, capsys, tmp_path):
        env_file = tmp_path / "github_env"
        monkeypatch.setenv("GITHUB_ENV", str(env_file))
        monkeypatch.setattr(probe_gpu_torch, "torch", _fake_torch())

        def _boom() -> None:
            raise RuntimeError(
                "no kernel image is available for execution on the device"
            )

        monkeypatch.setattr(probe_gpu_torch, "_run_op", _boom)
        assert main() == 0
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "sm_61" in out
        assert "no kernel image is available" in out
        assert env_file.read_text(encoding="utf-8") == "CUDA_VISIBLE_DEVICES=\n"
