import copy
import json
import sys
import warnings
from types import SimpleNamespace

import onyx
import paths
import pytest
import torch
import yaml
from common.game import GameState
from ml.dataset import convert_input_to_tensor
from ml.inference import TORCH
from ml.models.NorthernPenguin.model import (
    StateModelEncoder as RealStateModelEncoder,
)
from ml.validation.coverage.game_managers.model import process_game_manager
from onyx import create_torch_input, entrypoint, save_in_onnx

pytestmark = pytest.mark.integration

REFERENCE_GAMESTATES_DIR = paths.RESOURCES_PATH / "onnx" / "reference_gamestates"


class TestONNXConversion:
    def test_onnx_conversion_successful_on_real_randomized_model(
        self, tmp_path, game_states_fixture, config
    ):
        init_game_state: GameState = game_states_fixture[0]
        verification_game_states: list[GameState] = game_states_fixture[1:]

        with open(config) as file:
            model_kwargs = yaml.safe_load(file)

        mock_model_path = tmp_path / "real_random_model.pt"
        torch.save(
            obj=RealStateModelEncoder(**model_kwargs).state_dict(), f=mock_model_path
        )
        mock_onnx_output_path = tmp_path / "real_random_model.onnx"

        entrypoint(
            sample_gamestate=init_game_state,
            pytorch_model_path=mock_model_path,
            onnx_savepath=mock_onnx_output_path,
            model_def=RealStateModelEncoder,
            model_kwargs=model_kwargs,
            verification_gamestates=verification_game_states,
        )

    @pytest.fixture
    def game_states_fixture(self):
        json_files = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))
        return [_load_gamestate(path) for path in json_files]


class TestONNXExportWarnings:
    def test_export_emits_no_warnings(self, tmp_path, config):
        """save_in_onnx must not emit warnings from torch.onnx, PyG conv, or
        this repo (regression guard for #617)."""
        with open(config) as file:
            model_kwargs = yaml.safe_load(file)

        first = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]
        game_state = _load_gamestate(first)

        torch_model = RealStateModelEncoder(**model_kwargs).eval()
        hetero, _ = convert_input_to_tensor(game_state)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_in_onnx(torch_model, hetero, tmp_path / "model.onnx")

        offenders = [
            w
            for w in caught
            if (
                "torch/onnx/" in (w.filename or "")
                or "torch_geometric/nn/conv/" in (w.filename or "")
                or "/AIAgent/" in (w.filename or "")
            )
        ]
        assert not offenders, [
            f"{w.category.__name__} @ {w.filename}:{w.lineno}: {w.message}"
            for w in offenders
        ]


class TestForwardOnCUDA:
    """Regression guard for #617: StateModelEncoder.forward must run on CUDA.
    The v-v skip condition is a Python-int-derived tensor; if it is created
    without an explicit device it lands on CPU and torch.where raises
    "Expected all tensors to be on the same device" for CUDA inputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_forward_matches_cpu_with_and_without_v_v_edges(self, config):
        with open(config) as file:
            model_kwargs = yaml.safe_load(file)

        first = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]
        game_state = _load_gamestate(first)
        hetero, _ = convert_input_to_tensor(game_state)

        empty_hetero = copy.deepcopy(hetero)
        empty_hetero[TORCH.gamevertex_to_gamevertex].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        empty_hetero[TORCH.gamevertex_to_gamevertex].edge_type = torch.empty(
            (0,), dtype=torch.long
        )

        cpu_model = RealStateModelEncoder(**model_kwargs).eval()
        cuda_model = copy.deepcopy(cpu_model).cuda()

        for sample in (hetero, empty_hetero):
            args = create_torch_input(sample)
            expected = cpu_model(*args)
            cuda_args = tuple(arg.to("cuda") for arg in args)
            actual = cuda_model(*cuda_args)
            assert torch.allclose(expected, actual.cpu(), atol=1e-4)


class TestModelGamePreparatorKwargs:
    def test_create_onnx_model_uses_explicit_kwargs(
        self, tmp_path, monkeypatch, config
    ):
        with open(config) as file:
            model_kwargs = yaml.safe_load(file)

        captured = {}

        def fake_export(
            sample_gamestate,
            pytorch_model_path,
            onnx_savepath,
            model_def,
            model_kwargs,
            verification_gamestates=None,
        ):
            captured["model_kwargs"] = model_kwargs
            captured["pytorch_model_path"] = pytorch_model_path
            captured["onnx_savepath"] = onnx_savepath
            captured["model_def"] = model_def

        monkeypatch.setattr(
            process_game_manager, "save_torch_model_to_onnx_file", fake_export
        )
        model_path = tmp_path / "model.pth"
        onnx_path = tmp_path / "model.onnx"
        namespace = SimpleNamespace(
            shared_lock=None, is_prepared=SimpleNamespace(value=False)
        )
        preparator = process_game_manager.ModelGamePreparator(
            namespace,
            RealStateModelEncoder(**model_kwargs),
            model_path,
            onnx_path,
            model_kwargs,
        )
        preparator._create_onnx_model()

        assert captured["model_kwargs"] == model_kwargs
        assert captured["pytorch_model_path"] == model_path
        assert captured["onnx_savepath"] == onnx_path
        # The preparator resolves the model class through the registry
        # (name round-trip), not by reconstructing an import path.
        assert captured["model_def"] is RealStateModelEncoder


class TestMain:
    """CLI seam tests for ``onyx.main()``.

    ``main()`` is driven with a patched ``sys.argv`` and a recording fake for
    ``onyx.entrypoint`` so no export run happens; the model-kwargs YAML and the
    gamestate JSONs are the real golden inputs (the ``config`` fixture and the
    reference gamestates read through ``AIAgent/paths.py``).
    """

    def test_main_dispatches_to_entrypoint(self, tmp_path, monkeypatch, config):
        gamestates = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))
        sample, *verify = gamestates

        captured = _run_main(monkeypatch, _main_argv(tmp_path, config, sample, verify))

        assert set(captured) == {
            "sample_gamestate",
            "pytorch_model_path",
            "onnx_savepath",
            "model_def",
            "model_kwargs",
            "verification_gamestates",
        }
        assert captured["sample_gamestate"] == _load_gamestate(sample)
        assert captured["pytorch_model_path"] == tmp_path / "model.pth"
        assert captured["onnx_savepath"] == tmp_path / "model.onnx"
        assert captured["model_def"] is RealStateModelEncoder
        assert captured["model_kwargs"] == yaml.safe_load(config.read_text())
        assert captured["verification_gamestates"] == [
            _load_gamestate(path) for path in verify
        ]

    def test_main_without_verify_on_defaults_to_empty(
        self, tmp_path, monkeypatch, config
    ):
        sample = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]

        captured = _run_main(monkeypatch, _main_argv(tmp_path, config, sample, []))

        assert captured["verification_gamestates"] == []

    @pytest.mark.parametrize(
        "missing_flag",
        [
            "--sample-gamestate",
            "--pytorch-model",
            "--savepath",
            "--model-name",
            "--model-kwargs",
        ],
    )
    def test_main_missing_required_flag_exits(
        self, tmp_path, monkeypatch, config, missing_flag
    ):
        gamestates = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))
        sample, *verify = gamestates

        with pytest.raises(SystemExit):
            _run_main(
                monkeypatch,
                _main_argv(tmp_path, config, sample, verify, omit=missing_flag),
            )


class TestONNXVerification:
    """The verification failure path of ``entrypoint``.

    The current verification compares ``shorten_output`` results (rounded
    to 3 decimals, truncated to the first 10 elements) with an ``assert``;
    both weaknesses are #597-tracked. These tests pin the crash-level
    post-fix properties as strict ``xfail``s; #597 owns removing the
    markers when it lands.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#597: the verification compares shorten_output results with an "
            "assert, so a numerically divergent conversion surfaces as an "
            "AssertionError; #597's fix spec pins a proper (non-assert) "
            "error, so this pins that crash-level property"
        ),
    )
    def test_divergent_verification_raises_proper_error(
        self, tmp_path, monkeypatch, config
    ):
        with open(config) as file:
            model_kwargs = yaml.safe_load(file)

        gamestates = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))
        sample, *verify = gamestates

        mock_model_path = tmp_path / "real_random_model.pt"
        torch.save(
            obj=RealStateModelEncoder(**model_kwargs).state_dict(),
            f=mock_model_path,
        )

        real_onnx_run = onyx.onnx_run

        def divergent_onnx_run(session, data):
            (out,) = real_onnx_run(session, data)
            return [out + 1.0]

        monkeypatch.setattr(onyx, "onnx_run", divergent_onnx_run)

        with pytest.raises(Exception) as excinfo:
            entrypoint(
                sample_gamestate=_load_gamestate(sample),
                pytorch_model_path=mock_model_path,
                onnx_savepath=tmp_path / "real_random_model.onnx",
                model_def=RealStateModelEncoder,
                model_kwargs=model_kwargs,
                verification_gamestates=[_load_gamestate(path) for path in verify],
            )

        assert not isinstance(excinfo.value, AssertionError)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#597: the verification truncates both outputs to the first 10 "
            "elements, so a divergence beyond element 10 passes the current "
            "assert; #597's fix spec compares the full tensors with a proper "
            "(non-assert) error, so this pins that such a divergence is "
            "rejected with a proper error"
        ),
    )
    def test_divergence_beyond_first_ten_elements_is_rejected(
        self, tmp_path, monkeypatch, config
    ):
        with open(config) as file:
            model_kwargs = yaml.safe_load(file)

        sample = sorted(REFERENCE_GAMESTATES_DIR.glob("*.json"))[0]

        mock_model_path = tmp_path / "real_random_model.pt"
        torch.save(
            obj=RealStateModelEncoder(**model_kwargs).state_dict(),
            f=mock_model_path,
        )

        real_onnx_run = onyx.onnx_run

        def beyond_truncation_onnx_run(session, data):
            (out,) = real_onnx_run(session, data)
            perturbed = out.copy()
            perturbed.reshape(-1)[10] += 1.0
            return [perturbed]

        monkeypatch.setattr(onyx, "onnx_run", beyond_truncation_onnx_run)

        with pytest.raises(Exception) as excinfo:
            entrypoint(
                sample_gamestate=_load_gamestate(sample),
                pytorch_model_path=mock_model_path,
                onnx_savepath=tmp_path / "real_random_model.onnx",
                model_def=RealStateModelEncoder,
                model_kwargs=model_kwargs,
                verification_gamestates=[_twelve_state_gamestate()],
            )

        assert not isinstance(excinfo.value, AssertionError)


def _twelve_state_gamestate() -> GameState:
    """A 12-state verification gamestate built from the 86 reference.

    The reference has 4 states; 8 copies of its first state with fresh
    Ids (100-107) and empty ``Children`` bring it to 12, so the model
    output (one row per state) is longer than ``shorten_output``'s
    10-element truncation window.
    """
    path = REFERENCE_GAMESTATES_DIR / "86_gameState.json"
    with open(path) as f:
        data = json.load(f)
    template = data["States"][0]
    data["States"].extend({**template, "Id": 100 + i, "Children": []} for i in range(8))
    return GameState.from_dict(data)


def _run_main(monkeypatch, argv):
    """Run ``onyx.main()`` with the given argv; return the captured kwargs."""
    captured = {}

    def fake_entrypoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(onyx, "entrypoint", fake_entrypoint)
    monkeypatch.setattr(sys, "argv", argv)
    onyx.main()
    return captured


def _main_argv(tmp_path, config, sample, verify, omit=None):
    """The full ``onyx`` CLI argv; ``omit`` drops one required flag (and its value)."""
    flags = {
        "--sample-gamestate": [str(sample)],
        "--pytorch-model": [str(tmp_path / "model.pth")],
        "--savepath": [str(tmp_path / "model.onnx")],
        "--model-name": ["NorthernPenguin"],
        "--model-kwargs": [str(config)],
    }
    if verify:
        flags["--verify-on"] = [str(path) for path in verify]
    argv = ["onyx"]
    for flag, values in flags.items():
        if flag != omit:
            argv.extend([flag, *values])
    return argv


def _load_gamestate(path) -> GameState:
    with open(path) as f:
        game_state = GameState.from_dict(json.load(f))
        return game_state
