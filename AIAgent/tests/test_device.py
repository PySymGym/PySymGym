"""Tests for the device probe used by ``config.py`` and the e2e workflow."""

import torch

import device
from device import main, select_device


def _raise_no_kernel_image() -> None:
    raise RuntimeError(
        "CUDA error: no kernel image is available for execution on the device"
    )


class TestSelectDevice:
    def test_no_cuda_returns_cpu_silently(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert select_device() == torch.device("cpu")
        assert capsys.readouterr().out == ""

    def test_usable_gpu_returns_cuda_silently(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(device, "_smoke_kernel", lambda: None)
        assert select_device() == torch.device("cuda:0")
        assert capsys.readouterr().out == ""

    def test_incompatible_gpu_degrades_to_cpu_with_diagnosis(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(device, "_smoke_kernel", _raise_no_kernel_image)
        assert select_device() == torch.device("cpu")
        out = capsys.readouterr().out
        assert out.startswith("device=cpu (GPU ")
        assert f"not supported by torch {torch.__version__}" in out
        assert "no kernel image is available for execution on the device" in out


class TestMain:
    def test_no_cuda_verdict(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        main()
        assert capsys.readouterr().out.strip() == "device=cpu (no CUDA device)"

    def test_usable_gpu_verdict(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(device, "_smoke_kernel", lambda: None)
        monkeypatch.setattr(device, "_gpu_facts", lambda: "Fake GPU (sm_61)")
        main()
        assert (
            capsys.readouterr().out.strip()
            == f"device=cuda:0 (Fake GPU (sm_61), torch {torch.__version__})"
        )

    def test_incompatible_gpu_verdict(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(device, "_smoke_kernel", _raise_no_kernel_image)
        monkeypatch.setattr(
            device, "_gpu_facts", lambda: "NVIDIA GeForce GT 1030 (sm_61)"
        )
        monkeypatch.setattr(device, "_supported_archs", lambda: "sm_75 sm_80")
        main()
        assert capsys.readouterr().out.strip() == (
            "device=cpu (GPU NVIDIA GeForce GT 1030 (sm_61) not supported by torch "
            f"{torch.__version__} [supports: sm_75 sm_80]; "
            "CUDA error: no kernel image is available for execution on the device)"
        )
