"""The CUDA-saved checkpoint regression (#667), in the integration tier.

A checkpoint saved on CUDA must load on a CPU-only machine: without
``map_location`` the unpickler crashes with "Attempting to deserialize
object on a CUDA device but torch.cuda.is_available() is False". Creating
the reference checkpoint needs a real CUDA device, so the test carries the
``gpu`` marker (out of the machine-agnostic gated tier) and the
``cuda_is_usable`` skipif (the op-based verdict, as in
``probe_gpu_torch.py``, #662) for when it is selected explicitly on a
machine whose GPU the locked torch cannot run on. ``TestForwardOnCUDA`` in
``test_onnx.py`` is the precedent.

The load side exercises ``ml.model_io.safe_torch_load`` — the same helper
``run_training``'s ``model_init`` uses; the unit suite pins that
``model_init`` routes the downloaded checkpoint through it with
``map_location`` set to the training device.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from ml.device import cuda_is_usable, get_device
from ml.model_io import safe_torch_load
from ml.models.NorthernPenguin.model import StateModelEncoder

pytestmark = pytest.mark.integration

# The same model shape the unit suite's TestModelInit uses, so the
# checkpoint this test saves has the production model's parameter layout.
MODEL_KWARGS: dict[str, Any] = dict(
    hidden_channels=64,
    num_of_state_features=8,
    num_hops_1=2,
    num_hops_2=2,
    normalization=True,
    num_pc_layers=1,
)


@pytest.mark.gpu
class TestCudaSavedCheckpoint:
    @pytest.mark.skipif(not cuda_is_usable(), reason="requires usable CUDA")
    def test_cuda_saved_checkpoint_loads_on_cpu_only_machine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Save with CUDA visible (the autouse cpu_device fixture hides it);
        # the checkpoint pickle then records the cuda:0 location.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        reference = StateModelEncoder(**MODEL_KWARGS).to("cuda:0")
        path = tmp_path / "model.pth"
        torch.save(reference.state_dict(), path)
        # The load side must behave as on a CPU-only machine.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        model = StateModelEncoder(**MODEL_KWARGS)
        model.load_state_dict(safe_torch_load(path, get_device()))

        saved = reference.state_dict()
        for name, param in model.state_dict().items():
            assert param.device.type == "cpu"
            assert torch.equal(param, saved[name].cpu())
