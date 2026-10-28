"""Seam tests for the ML training entrypoint ``run_training.py``.

The entrypoint is driven with recording fakes — no end-to-end training runs:
``run_training()`` is called once with fakes for ``TrainingDataset``,
``optuna``, ``mlflow``, ``joblib`` and ``objective``, and the ``objective``
partial it wires up is inspected to extract the ``model_init`` / ``validate``
/ ``criterion_init`` closures, which are then driven directly. ``objective``
and ``main`` are tested the same way. The fakes pin the current (pre-#578)
behaviour: exact dispatch arguments, metric names, artifact layout and error
messages.
"""

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import run_training as run_training_module
from common.config.optuna_config import OptimizationDirection, OptunaConfig
from common.config.training_config import TrainingConfig
from common.config.validation_config import CriterionValidation, ValidationConfig
from ml.models.NorthernPenguin.model import StateModelEncoder
from paths import REPORT_PATH

pytestmark = pytest.mark.unit

# Evaluated at collection time, before the autouse cpu_device fixture patches
# torch.cuda.is_available: the real capability of the machine running the
# suite (the CUDA regression test for #667 needs it).
_REAL_CUDA_AVAILABLE = torch.cuda.is_available()

MODEL_KWARGS = dict(
    hidden_channels=64,
    num_of_state_features=8,
    num_hops_1=2,
    num_hops_2=2,
    normalization=True,
    num_pc_layers=1,
)


class FakeTrainingDataset:
    """A ``TrainingDataset`` stand-in recording construction and mode calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.switch_to_calls: list = []
        self.update_meta_data_calls = 0

    def switch_to(self, mode) -> None:
        self.switch_to_calls.append(mode)

    def update_meta_data(self) -> None:
        self.update_meta_data_calls += 1


class FakeMlflow:
    """A recording stand-in for the ``mlflow`` module surface used here."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.download_artifacts_result: Path | None = None

    def record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def called(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for (call, args, kwargs) in self.calls if call == name]

    def start_run(self, *args: Any, **kwargs: Any):
        self.record("start_run", *args, **kwargs)
        return contextlib.nullcontext()

    def last_active_run(self):
        self.record("last_active_run")
        return SimpleNamespace(info=SimpleNamespace(run_id="fake-run-id"))

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        self.record("log_artifact", *args, **kwargs)

    def set_tag(self, *args: Any, **kwargs: Any) -> None:
        self.record("set_tag", *args, **kwargs)

    def set_tracking_uri(self, *args: Any, **kwargs: Any) -> None:
        self.record("set_tracking_uri", *args, **kwargs)

    def set_experiment(self, *args: Any, **kwargs: Any) -> None:
        self.record("set_experiment", *args, **kwargs)

    def set_experiment_tags(self, *args: Any, **kwargs: Any) -> None:
        self.record("set_experiment_tags", *args, **kwargs)

    def enable_system_metrics_logging(self, *args: Any, **kwargs: Any) -> None:
        self.record("enable_system_metrics_logging", *args, **kwargs)

    def log_params(self, *args: Any, **kwargs: Any) -> None:
        self.record("log_params", *args, **kwargs)

    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        self.record("log_metrics", *args, **kwargs)

    @property
    def artifacts(self):
        return SimpleNamespace(download_artifacts=self._download_artifacts)

    def _download_artifacts(self, *args: Any, **kwargs: Any) -> Path:
        self.record("artifacts.download_artifacts", *args, **kwargs)
        return self.download_artifacts_result


class FakeJoblib:
    """A recording stand-in for the ``joblib`` module surface used here."""

    def __init__(self) -> None:
        self.dump_calls: list[tuple[Any, Any]] = []
        self.load_calls: list[Any] = []
        self.load_result = "loaded-trial-sentinel"

    def dump(self, obj: Any, path: Any) -> None:
        self.dump_calls.append((obj, path))

    def load(self, path: Any) -> Any:
        self.load_calls.append(path)
        return self.load_result


class FakeStudy:
    """An ``optuna.Study`` stand-in mirroring the per-trial callback contract."""

    def __init__(self) -> None:
        self.optimize_calls: list[dict] = []
        self.best_trial = SimpleNamespace(number=3)

    def optimize(self, func, **kwargs: Any) -> None:
        self.optimize_calls.append({"func": func, **kwargs})
        for callback in kwargs.get("callbacks") or []:
            callback(self, SimpleNamespace(number=0))


class FakeOptuna:
    """A stand-in for the ``optuna`` module surface used by ``run_training``."""

    def __init__(self) -> None:
        self.tpe_sampler_calls: list[dict] = []
        self.create_study_calls: list[dict] = []
        self.study = FakeStudy()
        self.samplers = SimpleNamespace(TPESampler=self._tpe_sampler)

    def _tpe_sampler(self, **kwargs: Any):
        self.tpe_sampler_calls.append(kwargs)
        return "tpe-sampler-sentinel"

    def create_study(self, **kwargs: Any) -> FakeStudy:
        self.create_study_calls.append(kwargs)
        return self.study


def make_objective_recorder():
    """A ``run_training.objective`` stand-in recording every call."""
    calls: list[tuple[Any, dict]] = []

    def objective(trial, **kwargs: Any) -> str:
        calls.append((trial, kwargs))
        return "objective-result-sentinel"

    objective.calls = calls
    return objective


def run_training_once(
    monkeypatch: pytest.MonkeyPatch,
    *,
    optuna_config: OptunaConfig,
    training_config: TrainingConfig,
    validation_config: ValidationConfig,
    weights_uri: str | None = None,
):
    """Drive ``run_training.run_training`` once with recording fakes.

    Returns ``(captured, fakes)``: ``captured.partial`` is the ``objective``
    partial in the study branch (``None`` in the trial-replay branch),
    ``captured.objective_calls`` the recorded ``objective`` calls (trial-replay
    branch only); ``fakes`` holds every recording fake.
    """
    seed_calls: list[int] = []
    monkeypatch.setattr(
        run_training_module, "seed_everything", lambda seed: seed_calls.append(seed)
    )
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    datasets: list[FakeTrainingDataset] = []

    def dataset_factory(*args: Any, **kwargs: Any) -> FakeTrainingDataset:
        dataset = FakeTrainingDataset(*args, **kwargs)
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(run_training_module, "TrainingDataset", dataset_factory)

    optuna = FakeOptuna()
    monkeypatch.setattr(run_training_module, "optuna", optuna)

    mlflow = FakeMlflow()
    monkeypatch.setattr(run_training_module, "mlflow", mlflow)

    joblib = FakeJoblib()
    monkeypatch.setattr(run_training_module, "joblib", joblib)

    objective = make_objective_recorder()
    monkeypatch.setattr(run_training_module, "objective", objective)

    run_training_module.run_training(
        optuna_config=optuna_config,
        training_config=training_config,
        validation_config=validation_config,
        weights_uri=weights_uri,
    )

    captured = SimpleNamespace(
        partial=(
            optuna.study.optimize_calls[0]["func"]
            if optuna.study.optimize_calls
            else None
        ),
        objective_calls=objective.calls,
    )
    fakes = SimpleNamespace(
        seed_calls=seed_calls,
        datasets=datasets,
        optuna=optuna,
        mlflow=mlflow,
        joblib=joblib,
    )
    return captured, fakes


def objective_keywords(captured) -> dict:
    """The keyword arguments the ``objective`` partial is wired with."""
    if captured.partial is not None:
        return captured.partial.keywords
    return captured.objective_calls[0][1]


def default_optuna_config(**overrides: Any) -> OptunaConfig:
    kwargs: dict[str, Any] = dict(
        n_startup_trials=1,
        n_trials=1,
        n_jobs=1,
        study_direction=OptimizationDirection.MAXIMIZE,
    )
    kwargs.update(overrides)
    return OptunaConfig(**kwargs)


def default_training_config(**overrides: Any) -> TrainingConfig:
    kwargs: dict[str, Any] = dict(
        dynamic_dataset=False,
        train_percentage=1.0,
        threshold_coverage=100,
        load_to_cpu=False,
        epochs=1,
        seed=1,
    )
    kwargs.update(overrides)
    return TrainingConfig(**kwargs)


def criterion_validation_config() -> ValidationConfig:
    return ValidationConfig(
        validation_mode=CriterionValidation(val_type="loss", batch_size=8)
    )


class FakeEncoder(nn.Module):
    """A ``StateModelEncoder`` stand-in recording construction and loads."""

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.loaded_state: dict | None = None

    def load_state_dict(self, state: dict) -> "FakeEncoder":
        self.loaded_state = state
        return self


class TestModelInit:
    """The ``model_init`` closure built inside ``run_training``."""

    def _model_init(
        self, monkeypatch: pytest.MonkeyPatch, weights_uri: str | None = None
    ):
        captured, fakes = run_training_once(
            monkeypatch,
            optuna_config=default_optuna_config(),
            training_config=default_training_config(),
            validation_config=criterion_validation_config(),
            weights_uri=weights_uri,
        )
        return objective_keywords(captured)["model_init"], fakes

    def test_weights_uri_none_returns_fresh_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_training_module, "StateModelEncoder", FakeEncoder)
        model_init, _ = self._model_init(monkeypatch, weights_uri=None)

        model = model_init(**MODEL_KWARGS)

        assert isinstance(model, FakeEncoder)
        assert model.params == MODEL_KWARGS
        assert model.loaded_state is None

    def test_weights_uri_loads_checkpoint_with_map_location(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(run_training_module, "StateModelEncoder", FakeEncoder)
        load_calls: list[tuple[Any, dict]] = []
        state = {"weight": torch.zeros(2)}
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, **kwargs: (load_calls.append((path, kwargs)), state)[1],
        )
        model_init, fakes = self._model_init(
            monkeypatch, weights_uri="mlflow-artifacts:/1/run/artifacts/model.pth"
        )
        fakes.mlflow.download_artifacts_result = tmp_path / "model.pth"

        model = model_init(**MODEL_KWARGS)

        # The trial-replay branch (weights_uri set, trial_uri None) also
        # downloads the trial artifact; the weights download is the call
        # carrying the artifact_uri keyword.
        weights_downloads = [
            (args, kwargs)
            for (args, kwargs) in fakes.mlflow.called("artifacts.download_artifacts")
            if "artifact_uri" in kwargs
        ]
        assert weights_downloads == [
            (
                (),
                {
                    "artifact_uri": "mlflow-artifacts:/1/run/artifacts/model.pth",
                    "dst_path": REPORT_PATH,
                },
            )
        ]
        # The #667 fix: the checkpoint is mapped to the device we train on
        # (cpu under the autouse cpu_device fixture), never left to the
        # device it was saved on.
        assert load_calls == [
            (tmp_path / "model.pth", {"map_location": torch.device("cpu")})
        ]
        assert model.loaded_state is state

    def test_cpu_saved_checkpoint_is_applied(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        model_init, fakes = self._model_init(
            monkeypatch, weights_uri="mlflow-artifacts:/1/run/artifacts/model.pth"
        )
        reference = StateModelEncoder(**MODEL_KWARGS)
        path = tmp_path / "model.pth"
        torch.save(reference.state_dict(), path)
        fakes.mlflow.download_artifacts_result = path

        model = model_init(**MODEL_KWARGS)

        loaded = model.state_dict()
        saved = reference.state_dict()
        assert set(loaded) == set(saved)
        for name in saved:
            assert torch.equal(loaded[name], saved[name])

    @pytest.mark.skipif(not _REAL_CUDA_AVAILABLE, reason="requires CUDA")
    def test_cuda_saved_checkpoint_loads_on_cpu_only_machine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Regression for #667: a CUDA-saved checkpoint must load when the
        # machine reports no CUDA. Without map_location this crashes with
        # "Attempting to deserialize object on a CUDA device but
        # torch.cuda.is_available() is False".
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        reference = StateModelEncoder(**MODEL_KWARGS).to("cuda:0")
        path = tmp_path / "model.pth"
        torch.save(reference.state_dict(), path)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        model_init, fakes = self._model_init(
            monkeypatch, weights_uri="mlflow-artifacts:/1/run/artifacts/model.pth"
        )
        fakes.mlflow.download_artifacts_result = path

        model = model_init(**MODEL_KWARGS)

        saved = reference.state_dict()
        for name, param in model.state_dict().items():
            assert param.device.type == "cpu"
            assert torch.equal(param, saved[name].cpu())
