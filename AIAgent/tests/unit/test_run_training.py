"""Seam tests for the ML training entrypoint ``run_training.py``.

The entrypoint is driven with recording fakes — no end-to-end training runs:
``run_training()`` is called once with fakes for ``TrainingDataset``,
``optuna``, ``mlflow``, ``joblib`` and ``objective``, and the ``objective``
partial it wires up is inspected to extract the ``model_init`` / ``validate``
/ ``criterion_init`` closures, which are then driven directly. ``objective``
and ``main`` are tested the same way. The fakes pin the post-#600 behaviour:
exact dispatch arguments, metric names, artifact layout, the best-checkpoint
export, the per-trial run-handle hand-off to the study-save callback, and
error messages.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError
from torch import nn

import run_training as run_training_module
from common.classes import GameFailed, GameResult, Map2Result
from common.config.broker_config import BrokerConfig
from common.config.connector_config import GameServerConnectorConfig
from common.config.dataset_config import DatasetConfig
from common.config.feature_config import FeatureConfig
from common.config.optuna_config import OptimizationDirection, OptunaConfig
from common.config.platform_config import Platform
from common.config.training_config import TrainingConfig
from common.config.validation_config import (
    CriterionValidation,
    CustomValidation,
    SVMValidationSendEachStep,
    ValidationConfig,
)
from common.dataset import TrainingDatasetMode
from common.validation_coverage.svm_info import SVMInfo
from ml.models.NorthernPenguin.model import StateModelEncoder
from ml.models.registry import get_model_class
from ml.training.seeds import derive_trial_seed
from ml.validation.statistics import AVERAGE_COVERAGE
from paths import (
    CURRENT_STUDY_PATH,
    CURRENT_TRIAL_PATH,
    MODEL_FILE_NAME,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    REPORT_PATH,
)
from run_training import BEST_TRIAL_NUMBER_TAG

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


class FakeRun:
    """A context-manager stand-in for the run ``mlflow.start_run`` returns."""

    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self) -> "FakeRun":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False


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
        return FakeRun("fake-run-id")

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
        # A frozen-trial stand-in: the replay branch reads its params.
        self.load_result = SimpleNamespace(params={"lr": 1e-4, "batch_size": 16})

    def dump(self, obj: Any, path: Any) -> None:
        self.dump_calls.append((obj, path))

    def load(self, path: Any) -> Any:
        self.load_calls.append(path)
        return self.load_result


class FakeStudy:
    """An ``optuna.Study`` stand-in mirroring the per-trial callback contract."""

    def __init__(self) -> None:
        self.optimize_calls: list[dict] = []
        self.enqueue_calls: list[dict] = []
        self.best_trial = SimpleNamespace(number=3)
        self.callback_trial = SimpleNamespace(number=0)

    def enqueue_trial(self, params: dict) -> None:
        self.enqueue_calls.append({"params": params})

    def optimize(self, func, **kwargs: Any) -> None:
        self.optimize_calls.append({"func": func, **kwargs})
        # Mirror Optuna's per-trial contract: the objective runs once per
        # trial, then the callbacks fire with the finished trial.
        for _ in range(kwargs.get("n_trials", 1)):
            func(self.callback_trial)
        for callback in kwargs.get("callbacks") or []:
            callback(self, self.callback_trial)


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
    download_artifacts_result: Path | None = None,
    broker_config: BrokerConfig | None = None,
    connector_config: GameServerConnectorConfig | None = None,
    feature_config: FeatureConfig | None = None,
):
    """Drive ``run_training.run_training`` once with recording fakes.

    Returns ``(captured, fakes)``: ``captured.partial`` is the ``objective``
    partial the study optimises (both branches), ``captured.objective_calls``
    the ``objective`` calls the fake study drove; ``fakes`` holds every
    recording fake. The sub-configs default to fresh models with the
    production defaults.
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
    mlflow.download_artifacts_result = download_artifacts_result
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
        broker_config=broker_config or BrokerConfig(),
        connector_config=connector_config or GameServerConnectorConfig(),
        feature_config=feature_config or FeatureConfig(),
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


def objective_config(captured):
    """The ``ObjectiveConfig`` the ``objective`` partial is wired with."""
    return captured.partial.keywords["config"]


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


class TestModelSelection:
    """The entrypoint's model is selected through the registry (#581)."""

    def test_module_model_is_the_registry_lookup(self):
        assert run_training_module.MODEL_NAME == "NorthernPenguin"
        assert run_training_module.StateModelEncoder is get_model_class(
            run_training_module.MODEL_NAME
        )


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
        return objective_config(captured).model_init, fakes

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


def svm_validation_config(fail_immediately: bool = True) -> ValidationConfig:
    platform = Platform(
        name="dotnet",
        DatasetConfigs=[
            DatasetConfig(dataset_base_path="/dummy", dataset_description="/dummy.json")
        ],
        SVMSInfo=[
            SVMInfo(
                name="VSharp",
                launch_command="run --port {port}",
                server_working_dir="/srv",
                min_port=35000,
                max_port=35050,
            )
        ],
    )
    return ValidationConfig(
        validation_mode=SVMValidationSendEachStep(
            val_type="svms_each_step",
            process_count=1,
            fail_immediately=fail_immediately,
            PlatformsConfig=[platform],
        )
    )


class TestRunTrainingStudyBranch:
    """``trial_uri is None and weights_uri is None``: tune a fresh study."""

    def test_dispatches_to_optuna_study(self, monkeypatch: pytest.MonkeyPatch) -> None:
        optuna_config = default_optuna_config(n_trials=2, n_startup_trials=3)
        training_config = default_training_config(
            train_percentage=0.8,
            threshold_steps_number=50,
            load_to_cpu=True,
            epochs=3,
            seed=42,
        )
        captured, fakes = run_training_once(
            monkeypatch,
            optuna_config=optuna_config,
            training_config=training_config,
            validation_config=criterion_validation_config(),
        )

        # Study-level seed, applied before the dataset is built.
        assert fakes.seed_calls == [42]

        # The dataset is built from the module path constants with the
        # training-config knobs.
        assert len(fakes.datasets) == 1
        dataset = fakes.datasets[0]
        assert dataset.init_args == (RAW_DATASET_PATH, PROCESSED_DATASET_PATH)
        assert dataset.init_kwargs == {
            "train_percentage": 0.8,
            "threshold_steps_number": 50,
            "load_to_cpu": True,
            "threshold_coverage": 100,
        }

        # Sampler and study creation.
        assert fakes.optuna.tpe_sampler_calls == [{"n_startup_trials": 3, "seed": 42}]
        assert fakes.optuna.create_study_calls == [
            {"sampler": "tpe-sampler-sentinel", "direction": "maximize"}
        ]

        # The study optimises the objective partial.
        assert len(fakes.optuna.study.optimize_calls) == 1
        optimize = fakes.optuna.study.optimize_calls[0]
        assert optimize["n_trials"] == 2
        assert optimize["gc_after_trial"] is True
        assert optimize["n_jobs"] == 1
        assert len(optimize["callbacks"]) == 1

        # The partial is wired with a single config object bundling the
        # run's dependencies.
        config = objective_config(captured)
        assert config.dataset is dataset
        assert config.dynamic_dataset is False
        assert config.epochs == 3
        assert config.direction is OptimizationDirection.MAXIMIZE
        assert config.seed == 42
        assert callable(config.model_init)
        assert callable(config.criterion_init)
        assert callable(config.validate)
        assert config.run_ids == {}

        # The criterion validation mode enables expandable segments.
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    def test_save_study_and_trial_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, fakes = run_training_once(
            monkeypatch,
            optuna_config=default_optuna_config(),
            training_config=default_training_config(),
            validation_config=criterion_validation_config(),
        )
        study = fakes.optuna.study

        # The fake study mirrors Optuna's per-trial callback contract: the
        # callback dumps study and trial. The fake objective never starts a
        # run, so no run handle is registered and the MLflow part is skipped
        # (the guard: no last_active_run guesswork, no foreign run).
        assert fakes.joblib.dump_calls == [
            (study, CURRENT_STUDY_PATH),
            (study.callback_trial, CURRENT_TRIAL_PATH),
        ]
        assert fakes.mlflow.called("start_run") == []
        assert fakes.mlflow.called("log_artifact") == []
        assert fakes.mlflow.called("set_tag") == []

    def test_save_study_and_trial_logs_to_the_registered_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured, fakes = run_training_once(
            monkeypatch,
            optuna_config=default_optuna_config(),
            training_config=default_training_config(),
            validation_config=criterion_validation_config(),
        )
        study = fakes.optuna.study
        # Register the run handle the way objective does, then fire the
        # callback for the trial that owns it.
        objective_config(captured).run_ids[study.callback_trial.number] = (
            "registered-run-id"
        )
        callback = study.optimize_calls[0]["callbacks"][0]
        callback(study, study.callback_trial)

        # The callback re-enters the trial's own run, logs both artifacts
        # and tags the best trial number so far.
        assert fakes.mlflow.called("start_run") == [(("registered-run-id",), {})]
        assert fakes.mlflow.called("log_artifact") == [
            ((CURRENT_STUDY_PATH,), {}),
            ((CURRENT_TRIAL_PATH,), {}),
        ]
        assert fakes.mlflow.called("set_tag") == [
            ((BEST_TRIAL_NUMBER_TAG, study.best_trial.number), {})
        ]


class TestRunTrainingTrialReplayBranch:
    """``trial_uri``/``weights_uri`` set: replay the downloaded trial."""

    def test_replays_downloaded_trial(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        trial_path = tmp_path / "trial.pkl"
        optuna_config = default_optuna_config(
            n_trials=2, trial_uri="mlflow-artifacts:/1/run/artifacts/trial.pkl"
        )
        captured, fakes = run_training_once(
            monkeypatch,
            optuna_config=optuna_config,
            training_config=default_training_config(),
            validation_config=criterion_validation_config(),
            weights_uri="mlflow-artifacts:/1/run/artifacts/model.pth",
            download_artifacts_result=trial_path,
        )
        study = fakes.optuna.study

        # The trial artifact is downloaded (uri positionally, dst_path as a
        # string) and loaded with joblib.
        assert fakes.mlflow.called("artifacts.download_artifacts") == [
            (
                ("mlflow-artifacts:/1/run/artifacts/trial.pkl",),
                {"dst_path": str(REPORT_PATH)},
            )
        ]
        assert fakes.joblib.load_calls == [trial_path]

        # The replay runs through a fresh study, like a tuning run: the
        # same sampler and direction, the frozen trial's params enqueued
        # once per replay, one optimize with the shared per-trial
        # persistence.
        assert fakes.optuna.create_study_calls == [
            {"sampler": "tpe-sampler-sentinel", "direction": "maximize"}
        ]
        assert study.enqueue_calls == [
            {"params": {"lr": 1e-4, "batch_size": 16}},
            {"params": {"lr": 1e-4, "batch_size": 16}},
        ]
        assert len(study.optimize_calls) == 1
        optimize = study.optimize_calls[0]
        assert optimize["func"] is captured.partial
        assert optimize["n_trials"] == 2
        assert optimize["gc_after_trial"] is True
        assert optimize["n_jobs"] == 1
        assert len(optimize["callbacks"]) == 1

        # The objective is driven n_trials times with the shared config.
        config = objective_config(captured)
        assert len(captured.objective_calls) == 2
        for trial, kwargs in captured.objective_calls:
            assert trial is study.callback_trial
            assert kwargs == {"config": config}
        assert config.dataset is fakes.datasets[0]
        assert callable(config.model_init)
        assert callable(config.validate)

        # The callback fired with no registered run (the fake objective
        # never starts one): the pickles are dumped, no MLflow calls.
        assert fakes.joblib.dump_calls == [
            (study, CURRENT_STUDY_PATH),
            (study.callback_trial, CURRENT_TRIAL_PATH),
        ]
        assert fakes.mlflow.called("start_run") == []
        assert fakes.mlflow.called("log_artifact") == []
        assert fakes.mlflow.called("set_tag") == []


class FakeValidationCoverage:
    """A ``ValidationCoverage`` stand-in recording construction and calls."""

    def __init__(
        self,
        model: Any,
        dataset: Any,
        *,
        broker_config: Any = None,
        connector_config: Any = None,
        feature_config: Any = None,
        model_kwargs: dict | None = None,
        trial_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.broker_config = broker_config
        self.connector_config = connector_config
        self.feature_config = feature_config
        self.model_kwargs = model_kwargs
        self.trial_dir = trial_dir
        self.validate_coverage_calls: list[tuple] = []
        self.result: list = []

    def validate_coverage(self, maps: list, validation_config: Any, **kwargs: Any):
        self.validate_coverage_calls.append((maps, validation_config))
        return self.result


class TestValidateClosures:
    """The ``validate`` closures built by ``get_validation``."""

    def _validate(
        self, monkeypatch: pytest.MonkeyPatch, validation_config: ValidationConfig
    ):
        captured, fakes = run_training_once(
            monkeypatch,
            optuna_config=default_optuna_config(),
            training_config=default_training_config(),
            validation_config=validation_config,
        )
        return objective_config(captured).validate, fakes

    def test_criterion_validate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        validate, fakes = self._validate(monkeypatch, criterion_validation_config())
        loss_calls: list[tuple] = []
        monkeypatch.setattr(
            run_training_module,
            "validate_loss",
            lambda model, dataset, criterion, batch_size: (
                loss_calls.append((model, dataset, criterion, batch_size)),
                0.123,
            )[1],
        )
        model = object()
        dataset = fakes.datasets[0]

        result, metrics = validate(model, dataset, tmp_path, {})

        assert dataset.switch_to_calls == [TrainingDatasetMode.VALIDATION]
        assert len(loss_calls) == 1
        called_model, called_dataset, criterion, batch_size = loss_calls[0]
        assert called_model is model
        assert called_dataset is dataset
        assert batch_size == 8
        assert isinstance(criterion, nn.BCEWithLogitsLoss)
        assert criterion.reduction == "sum"
        assert result == 0.123
        # Metric name: str(criterion) with parentheses replaced, plus the
        # dataset mode value (torch 2.7.1 renders the criterion as
        # "BCEWithLogitsLoss()").
        assert metrics == {"BCEWithLogitsLoss__VALIDATION": 0.123}

    def test_criterion_validate_training_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        validation_config = ValidationConfig(
            validation_mode=CriterionValidation(
                val_type="loss", batch_size=8, dataset=TrainingDatasetMode.TRAINING
            )
        )
        validate, fakes = self._validate(monkeypatch, validation_config)
        monkeypatch.setattr(run_training_module, "validate_loss", lambda *a: 0.5)
        dataset = fakes.datasets[0]

        result, metrics = validate(object(), dataset, tmp_path, {})

        assert dataset.switch_to_calls == [TrainingDatasetMode.TRAINING]
        assert result == 0.5
        assert metrics == {"BCEWithLogitsLoss__TRAINING": 0.5}

    def test_svm_validate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, game_map2svm_factory
    ) -> None:
        maps = [game_map2svm_factory("MapA"), game_map2svm_factory("MapB")]
        monkeypatch.setattr(run_training_module, "get_maps", lambda config: maps)
        coverage_inits: list[FakeValidationCoverage] = []

        def coverage_factory(
            model,
            dataset,
            broker_config=None,
            connector_config=None,
            feature_config=None,
            model_kwargs=None,
            trial_dir=None,
        ):
            instance = FakeValidationCoverage(
                model,
                dataset,
                broker_config=broker_config,
                connector_config=connector_config,
                feature_config=feature_config,
                model_kwargs=model_kwargs,
                trial_dir=trial_dir,
            )
            coverage_inits.append(instance)
            return instance

        monkeypatch.setattr(run_training_module, "ValidationCoverage", coverage_factory)
        stats_calls: list[tuple] = []
        metrics = {AVERAGE_COVERAGE: 0.5, "extra": 1}
        monkeypatch.setattr(
            run_training_module,
            "get_svms_statistics",
            lambda results, config, dataset, table_path: (
                stats_calls.append((results, config, dataset, table_path)),
                metrics,
            )[1],
        )

        validation_config = svm_validation_config()
        validate, fakes = self._validate(monkeypatch, validation_config)
        model = object()
        dataset = fakes.datasets[0]
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()

        result, returned_metrics = validate(
            model, dataset, trial_dir, {"hidden_channels": 64}
        )

        # The SVM validation mode disables expandable segments (it does not
        # work with multiprocessing).
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"

        coverage = coverage_inits[0]
        assert coverage.model is model
        assert coverage.dataset is dataset
        # The run's sub-configs are threaded to the coverage validator.
        assert coverage.broker_config == BrokerConfig()
        assert coverage.connector_config == GameServerConnectorConfig()
        assert coverage.feature_config == FeatureConfig()
        assert coverage.model_kwargs == {"hidden_channels": 64}
        assert coverage.trial_dir == trial_dir
        assert coverage.validate_coverage_calls == [
            (maps, validation_config.validation_mode)
        ]

        table_path = trial_dir / "svms_result_table.csv"
        assert stats_calls == [
            (coverage.result, validation_config.validation_mode, dataset, table_path)
        ]
        # The study-branch callback also logs the study/trial pickles; the
        # validate call logs the per-trial statistics table.
        log_artifact_calls = fakes.mlflow.called("log_artifact")
        assert ((table_path,), {}) in log_artifact_calls
        assert result == metrics[AVERAGE_COVERAGE]
        assert returned_metrics is metrics

        # Per-trial statistics table: sorted map names as the header.
        assert table_path.read_text() == "MapA,MapB\n"

        # A second call must not rewrite the header.
        validate(model, dataset, trial_dir, {})
        assert table_path.read_text() == "MapA,MapB\n"

    def test_svm_validate_fail_immediately(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, game_map2svm_factory
    ) -> None:
        maps = [
            game_map2svm_factory("MapA"),
            game_map2svm_factory("MapB"),
            game_map2svm_factory("MapC"),
        ]
        monkeypatch.setattr(run_training_module, "get_maps", lambda config: maps)

        def coverage_factory(
            model,
            dataset,
            broker_config=None,
            connector_config=None,
            feature_config=None,
            model_kwargs=None,
            trial_dir=None,
        ):
            instance = FakeValidationCoverage(
                model,
                dataset,
                broker_config=broker_config,
                connector_config=connector_config,
                feature_config=feature_config,
                model_kwargs=model_kwargs,
                trial_dir=trial_dir,
            )
            # The failure correlation (#616): the aggregate error must name
            # every failed map with its reason; the successful map is not
            # mentioned.
            instance.result = [
                Map2Result(
                    maps[0],
                    GameFailed(
                        "[each-step] map MapA timed out after 1800.0s "
                        "(step budget 10) at step 0/10"
                    ),
                ),
                Map2Result(
                    maps[1],
                    GameResult(
                        steps_count=10,
                        tests_count=1,
                        errors_count=0,
                        actual_coverage_percent=100,
                    ),
                ),
                Map2Result(
                    maps[2],
                    GameFailed(
                        "Not all steps exhausted on MapC: 5 of 10 steps "
                        "taken, actual coverage 50.00% != 100%"
                    ),
                ),
            ]
            return instance

        monkeypatch.setattr(run_training_module, "ValidationCoverage", coverage_factory)
        monkeypatch.setattr(
            run_training_module,
            "get_svms_statistics",
            lambda *a: {AVERAGE_COVERAGE: 0.0},
        )

        validate, fakes = self._validate(
            monkeypatch, svm_validation_config(fail_immediately=True)
        )
        model = object()
        dataset = fakes.datasets[0]
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()

        with pytest.raises(RuntimeError) as excinfo:
            validate(model, dataset, trial_dir, {})

        assert str(excinfo.value) == (
            "Validation failed: "
            "MapA: [each-step] map MapA timed out after 1800.0s "
            "(step budget 10) at step 0/10; "
            "MapC: Not all steps exhausted on MapC: 5 of 10 steps "
            "taken, actual coverage 50.00% != 100%"
        )

    def test_svm_validate_failed_map_without_fail_immediately(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, game_map2svm_factory
    ) -> None:
        maps = [game_map2svm_factory("MapA")]
        monkeypatch.setattr(run_training_module, "get_maps", lambda config: maps)
        coverage_inits: list[FakeValidationCoverage] = []

        def coverage_factory(
            model,
            dataset,
            broker_config=None,
            connector_config=None,
            feature_config=None,
            model_kwargs=None,
            trial_dir=None,
        ):
            instance = FakeValidationCoverage(
                model,
                dataset,
                broker_config=broker_config,
                connector_config=connector_config,
                feature_config=feature_config,
                model_kwargs=model_kwargs,
                trial_dir=trial_dir,
            )
            instance.result = [
                Map2Result(maps[0], GameFailed(reason=Exception("boom")))
            ]
            coverage_inits.append(instance)
            return instance

        monkeypatch.setattr(run_training_module, "ValidationCoverage", coverage_factory)
        monkeypatch.setattr(
            run_training_module,
            "get_svms_statistics",
            lambda *a: {AVERAGE_COVERAGE: 0.0},
        )

        validate, fakes = self._validate(
            monkeypatch, svm_validation_config(fail_immediately=False)
        )
        model = object()
        dataset = fakes.datasets[0]
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()

        result, metrics = validate(model, dataset, trial_dir, {})

        assert result == 0.0
        assert metrics == {AVERAGE_COVERAGE: 0.0}

    def test_custom_validate_runs_pipeline_in_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, game_map2svm_factory
    ) -> None:
        criterion_mode = CriterionValidation(val_type="loss", batch_size=8)
        svm_config = svm_validation_config()
        custom = CustomValidation(
            val_type="custom",
            val_sequence=[criterion_mode, svm_config.validation_mode],
            process_count=1,
        )
        maps = [game_map2svm_factory("MapA")]
        monkeypatch.setattr(run_training_module, "get_maps", lambda config: maps)
        order: list[str] = []
        monkeypatch.setattr(
            run_training_module,
            "validate_loss",
            lambda *a: (order.append("criterion"), 0.1)[1],
        )
        coverage_inits: list[FakeValidationCoverage] = []

        def coverage_factory(
            model,
            dataset,
            broker_config=None,
            connector_config=None,
            feature_config=None,
            model_kwargs=None,
            trial_dir=None,
        ):
            instance = FakeValidationCoverage(
                model,
                dataset,
                broker_config=broker_config,
                connector_config=connector_config,
                feature_config=feature_config,
                model_kwargs=model_kwargs,
                trial_dir=trial_dir,
            )
            # A successful result: the default fail_immediately=True would
            # abort the pipeline on a GameFailed.
            instance.result = [
                Map2Result(
                    maps[0],
                    GameResult(
                        steps_count=1000,
                        tests_count=2,
                        errors_count=3,
                        actual_coverage_percent=100.0,
                    ),
                )
            ]
            coverage_inits.append(instance)
            return instance

        monkeypatch.setattr(run_training_module, "ValidationCoverage", coverage_factory)
        svm_metrics = {AVERAGE_COVERAGE: 0.5}
        monkeypatch.setattr(
            run_training_module,
            "get_svms_statistics",
            lambda *a: (order.append("svm"), svm_metrics)[1],
        )

        validate, fakes = self._validate(
            monkeypatch, ValidationConfig(validation_mode=custom)
        )
        model = object()
        dataset = fakes.datasets[0]
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()

        result, metrics = validate(model, dataset, trial_dir, {})

        assert order == ["criterion", "svm"]
        # The first stage's result is returned; the metrics are merged.
        assert result == 0.1
        assert metrics == {
            "BCEWithLogitsLoss__VALIDATION": 0.1,
            AVERAGE_COVERAGE: 0.5,
        }
        # The last stage (SVM) leaves expandable segments disabled.
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


class FakeTrial:
    """An ``optuna.Trial`` stand-in with a scripted search space."""

    FLOAT_VALUES = {"lr": 1e-4}
    INT_VALUES = {
        "batch_size": 16,
        "num_of_state_features": 8,
        "hidden_channels": 64,
        "num_hops_1": 2,
        "num_hops_2": 3,
        "num_pc_layers": 1,
    }

    def __init__(self, number: int = 7) -> None:
        self.number = number
        self.suggest_calls: list[tuple] = []

    def suggest_float(self, name: str, low: float, high: float) -> float:
        self.suggest_calls.append(("float", name, low, high))
        return self.FLOAT_VALUES[name]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.suggest_calls.append(("int", name, low, high))
        return self.INT_VALUES[name]


class RecordingModel(nn.Module):
    """A real ``nn.Module`` (one ``Linear``) with spied ``train``/``eval``/``to``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.mode_calls: list[str] = []
        self.to_calls: list[tuple] = []

    def train(self, mode: bool = True) -> "RecordingModel":
        # ``nn.Module.eval`` is ``train(False)`` in torch 2.7, so one spy
        # covers both mode switches.
        self.mode_calls.append("train" if mode else "eval")
        return super().train(mode)

    def to(self, *args: Any, **kwargs: Any) -> "RecordingModel":
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def make_fake_early_stopping(continue_results: list[bool]):
    """An ``EarlyStopping`` stand-in with a scripted ``is_continue``."""
    inits: list[dict] = []
    is_continue_calls: list = []

    class FakeEarlyStopping:
        def __init__(self, **kwargs: Any) -> None:
            inits.append(kwargs)

        def is_continue(self, state_value: Any) -> bool:
            is_continue_calls.append(state_value)
            return continue_results.pop(0)

    return FakeEarlyStopping, inits, is_continue_calls


class TestObjective:
    """The per-trial ``objective`` training loop."""

    EXPECTED_SEARCH_SPACE = [
        ("float", "lr", 1e-7, 1e-3),
        ("int", "batch_size", 8, 32),
        ("int", "num_of_state_features", 8, 64),
        ("int", "hidden_channels", 64, 128),
        ("int", "num_hops_1", 2, 10),
        ("int", "num_hops_2", 2, 10),
        ("int", "num_pc_layers", 1, 5),
    ]

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        epochs: int = 2,
        dynamic_dataset: bool = False,
        continue_results: list[bool] | None = None,
        validate_results: list[tuple] | None = None,
        seed: int = 1,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        trial_number: int = 7,
        mutate_on_train: bool = False,
    ):
        trial = FakeTrial(number=trial_number)
        dataset = FakeTrainingDataset()
        model = RecordingModel()
        model_init_calls: list[dict] = []

        def model_init(**kwargs: Any) -> RecordingModel:
            model_init_calls.append(kwargs)
            return model

        criterion_init_calls: list[str] = []

        def criterion_init() -> str:
            criterion_init_calls.append("criterion")
            return "criterion-sentinel"

        if continue_results is None:
            continue_results = [True] * epochs
        if validate_results is None:
            validate_results = [(10.0 + i, {"metric": 10.0 + i}) for i in range(epochs)]
        validate_calls: list[tuple] = []

        def validate(model_, dataset_, trial_directory, model_kwargs):
            validate_calls.append((model_, dataset_, trial_directory, model_kwargs))
            return validate_results.pop(0)

        seed_calls: list[int] = []
        monkeypatch.setattr(
            run_training_module, "seed_everything", lambda s: seed_calls.append(s)
        )
        early_stopping_cls, es_inits, es_is_continue_calls = make_fake_early_stopping(
            continue_results
        )
        monkeypatch.setattr(run_training_module, "EarlyStopping", early_stopping_cls)
        train_calls: list[dict] = []

        def fake_train(**kwargs: Any) -> None:
            if mutate_on_train:
                # Mark the model with the train-call index so each epoch's
                # weights differ and the restored checkpoint is identifiable.
                kwargs["model"].linear.bias.data.fill_(float(len(train_calls)))
            train_calls.append(kwargs)

        monkeypatch.setattr(run_training_module, "train", fake_train)
        dataloader_calls: list[tuple] = []

        def dataloader(dataset_, batch_size, **kwargs):
            dataloader_calls.append((dataset_, batch_size, kwargs))
            return "dataloader-sentinel"

        monkeypatch.setattr(run_training_module, "DataLoader", dataloader)
        monkeypatch.setattr(
            run_training_module, "trial_dir", lambda n: tmp_path / "trials" / str(n)
        )
        mlflow = FakeMlflow()
        monkeypatch.setattr(run_training_module, "mlflow", mlflow)

        run_ids: dict[int, str] = {}
        config = run_training_module.ObjectiveConfig(
            dataset=dataset,
            dynamic_dataset=dynamic_dataset,
            model_init=model_init,
            criterion_init=criterion_init,
            epochs=epochs,
            validate=validate,
            direction=direction,
            seed=seed,
            run_ids=run_ids,
        )
        result = run_training_module.objective(trial, config=config)

        return SimpleNamespace(
            result=result,
            trial=trial,
            dataset=dataset,
            model=model,
            model_init_calls=model_init_calls,
            criterion_init_calls=criterion_init_calls,
            seed_calls=seed_calls,
            es_inits=es_inits,
            es_is_continue_calls=es_is_continue_calls,
            train_calls=train_calls,
            dataloader_calls=dataloader_calls,
            validate_calls=validate_calls,
            mlflow=mlflow,
            run_ids=run_ids,
        )

    def test_full_loop(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rec = self._run(monkeypatch, tmp_path, epochs=2, dynamic_dataset=True)
        trial_dir = tmp_path / "trials" / "7"
        model_path = trial_dir / MODEL_FILE_NAME
        kwargs_path = trial_dir / "model_kwargs.yaml"

        # Per-trial seed: derived from the study seed and the trial number.
        assert rec.seed_calls == [derive_trial_seed(1, 7)]

        # The exact optuna search space.
        assert rec.trial.suggest_calls == self.EXPECTED_SEARCH_SPACE

        # Early stopping is constructed from the TrialSettings constants.
        assert rec.es_inits == [
            {
                "state_len": 5,
                "tolerance": 0.0001,
                "direction": OptimizationDirection.MAXIMIZE,
            }
        ]

        # model_init gets exactly the StateModelEncoder.__init__ kwargs,
        # drawn from the trial's suggestions.
        expected_kwargs = {
            "hidden_channels": 64,
            "num_of_state_features": 8,
            "num_hops_1": 2,
            "num_hops_2": 3,
            "normalization": True,
            "num_pc_layers": 1,
        }
        assert rec.model_init_calls == [expected_kwargs]
        # The criterion is created once per trial, not per epoch.
        assert rec.criterion_init_calls == ["criterion"]
        # The model is moved to the training device (cpu under the fixture).
        assert rec.model.to_calls == [((torch.device("cpu"),), {})]

        # Trial artifact layout.
        assert trial_dir.is_dir()
        assert yaml.safe_load(kwargs_path.read_text()) == expected_kwargs
        assert rec.mlflow.called("start_run") == [((), {"run_name": "7"})]
        # The trial's run handle is registered for the study-save callback.
        assert rec.run_ids == {7: "fake-run-id"}
        assert rec.mlflow.called("log_params") == [
            (
                (
                    {
                        "lr": 1e-4,
                        "batch_size": 16,
                        "num_of_state_features": 8,
                        "hidden_channels": 64,
                        "num_hops_1": 2,
                        "num_hops_2": 3,
                        "num_pc_layers": 1,
                        "normalization": True,
                        "early_stopping_state_len": 5,
                        "tolerance": 0.0001,
                    },
                ),
                {},
            )
        ]
        assert rec.mlflow.called("log_artifact") == [
            ((kwargs_path,), {}),
            ((model_path, "0"), {}),
            ((model_path, "1"), {}),
            # The final export re-logs the best checkpoint under the last
            # epoch's slot (here the best is the last epoch, so the weights
            # are unchanged).
            ((model_path, "1"), {}),
        ]

        # Per-epoch sequence.
        assert rec.dataset.switch_to_calls == [
            TrainingDatasetMode.TRAINING,
            TrainingDatasetMode.VALIDATION,
            TrainingDatasetMode.TRAINING,
            TrainingDatasetMode.VALIDATION,
        ]
        assert rec.dataloader_calls == [
            (rec.dataset, 16, {"shuffle": False}),
            (rec.dataset, 16, {"shuffle": False}),
        ]
        assert rec.model.mode_calls == ["train", "eval", "train", "eval"]
        assert len(rec.train_calls) == 2
        for call in rec.train_calls:
            assert call["dataloader"] == "dataloader-sentinel"
            assert call["model"] is rec.model
            assert isinstance(call["optimizer"], torch.optim.Adam)
            assert call["optimizer"].defaults["lr"] == 1e-4
            assert call["criterion"] == "criterion-sentinel"
        # The checkpoint is really saved each epoch, and the final export
        # restores the best checkpoint (here the best is the last epoch, so
        # the file equals the model's final state).
        assert model_path.exists()
        loaded = torch.load(model_path, weights_only=True)
        saved = rec.model.state_dict()
        assert set(loaded) == set(saved)
        for name in saved:
            assert torch.equal(loaded[name], saved[name])
        assert rec.validate_calls == [
            (rec.model, rec.dataset, trial_dir, expected_kwargs),
            (rec.model, rec.dataset, trial_dir, expected_kwargs),
        ]
        assert rec.mlflow.called("log_metrics") == [
            (({"metric": 10.0},), {"step": 0}),
            (({"metric": 11.0},), {"step": 1}),
        ]
        # dynamic_dataset=True refreshes the metadata after each validation.
        assert rec.dataset.update_meta_data_calls == 2
        assert rec.es_is_continue_calls == [10.0, 11.0]
        # The last validation result is returned.
        assert rec.result == 11.0

    def test_early_stop_breaks_and_prints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capfd: pytest.CaptureFixture,
    ) -> None:
        rec = self._run(
            monkeypatch,
            tmp_path,
            epochs=3,
            continue_results=[True, False, True],
            validate_results=[
                (1.0, {"m": 1.0}),
                (2.0, {"m": 2.0}),
                (3.0, {"m": 3.0}),
            ],
        )

        # The loop breaks right after the epoch that stopped continuing:
        # epochs 0 and 1 both trained, epoch 2 never started.
        assert len(rec.train_calls) == 2
        assert rec.es_is_continue_calls == [1.0, 2.0]
        assert rec.result == 2.0
        captured = capfd.readouterr()
        assert "Training was stopped on 1 epoch." in captured.out

    @pytest.mark.parametrize(
        ("direction", "results", "best"),
        [
            (OptimizationDirection.MAXIMIZE, [1.0, 3.0, 2.0], 3.0),
            (OptimizationDirection.MINIMIZE, [3.0, 1.0, 2.0], 1.0),
        ],
    )
    def test_restores_the_best_checkpoint_when_the_best_epoch_is_not_last(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        direction: OptimizationDirection,
        results: list[float],
        best: float,
    ) -> None:
        # Epoch 1 is the best; epoch 2 regresses and triggers the stop. The
        # exported model and the trial's value must be epoch 1's, not
        # epoch 2's.
        rec = self._run(
            monkeypatch,
            tmp_path,
            epochs=3,
            continue_results=[True, True, False],
            validate_results=[(value, {"m": value}) for value in results],
            direction=direction,
            mutate_on_train=True,
        )
        model_path = tmp_path / "trials" / "7" / MODEL_FILE_NAME

        # The fake train marks the bias with the call index: the restored
        # checkpoint carries epoch 1's mark (1.0), not epoch 2's (2.0).
        loaded = torch.load(model_path, weights_only=True)
        assert loaded["linear.bias"].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert rec.model.linear.bias.tolist() == [1.0, 1.0, 1.0, 1.0]
        assert rec.result == best
        # The final export re-logs the best checkpoint under the last
        # epoch's slot.
        assert rec.mlflow.called("log_artifact")[-1] == ((model_path, "2"), {})

    def test_static_dataset_never_updates_meta_data(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rec = self._run(monkeypatch, tmp_path, epochs=2, dynamic_dataset=False)

        assert rec.dataset.update_meta_data_calls == 0
        assert rec.result == 11.0


def default_main_config() -> dict:
    """A production-shaped config (alias keys) for the ``main`` tests."""
    return {
        "OptunaConfig": {
            "n_startup_trials": 1,
            "n_trials": 1,
            "n_jobs": 1,
            "study_direction": "maximize",
        },
        "TrainingConfig": {
            "dynamic_dataset": False,
            "train_percentage": 1.0,
            "threshold_coverage": 100,
            "load_to_cpu": False,
            "epochs": 1,
            "seed": 99,
        },
        "ValidationConfig": {
            "validation_mode": {"val_type": "loss", "batch_size": 8},
        },
        "MLFlowConfig": {"experiment_name": "CI"},
    }


class TestMain:
    """``run_training.main``: config loading and MLflow setup."""

    def _main(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict):
        # The log path is anchored to the AIAgent directory (paths.py); point
        # it at tmp so the test leaves no artifact behind.
        log_path = tmp_path / "ml_app.log"
        monkeypatch.setattr(run_training_module, "LOG_PATH", log_path)
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.safe_dump(config))

        run_training_calls: list[dict] = []
        monkeypatch.setattr(
            run_training_module,
            "run_training",
            lambda **kwargs: run_training_calls.append(kwargs),
        )
        set_start_method_calls: list[tuple] = []
        monkeypatch.setattr(
            run_training_module.mp,
            "set_start_method",
            lambda *args, **kwargs: set_start_method_calls.append((args, kwargs)),
        )
        create_folder_calls: list[list] = []
        monkeypatch.setattr(
            run_training_module,
            "create_folders_if_necessary",
            lambda folders: create_folder_calls.append(folders),
        )
        create_file_calls: list[Path] = []
        monkeypatch.setattr(
            run_training_module,
            "create_file",
            lambda path: create_file_calls.append(path),
        )
        mlflow = FakeMlflow()
        monkeypatch.setattr(run_training_module, "mlflow", mlflow)

        gpu_logger = logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor")
        original_level = gpu_logger.level
        gpu_level: int | None = None
        try:
            run_training_module.main(str(config_path))
            gpu_level = gpu_logger.level
        finally:
            gpu_logger.setLevel(original_level)

        return SimpleNamespace(
            run_training_calls=run_training_calls,
            set_start_method_calls=set_start_method_calls,
            create_folder_calls=create_folder_calls,
            create_file_calls=create_file_calls,
            log_path=log_path,
            mlflow=mlflow,
            gpu_level=gpu_level,
        )

    def test_main_without_tracking_uri(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capfd: pytest.CaptureFixture,
    ) -> None:
        rec = self._main(monkeypatch, tmp_path, default_main_config())

        # Spawn start method, the processed-dataset folder, the log file.
        assert rec.set_start_method_calls == [(("spawn",), {"force": True})]
        assert rec.create_folder_calls == [[PROCESSED_DATASET_PATH]]
        assert rec.create_file_calls == [rec.log_path]

        # The training device is printed (cpu under the fixture).
        assert capfd.readouterr().out == "cpu\n"

        # MLflow setup: no tracking uri, experiment, tags, system metrics.
        assert rec.mlflow.called("set_tracking_uri") == []
        assert rec.mlflow.called("set_experiment") == [(("CI",), {})]
        assert rec.mlflow.called("enable_system_metrics_logging") == [((), {})]
        # The gpu-monitor logger is silenced to ERROR.
        assert rec.gpu_level == logging.ERROR

        # run_training gets the parsed config.
        assert len(rec.run_training_calls) == 1
        call = rec.run_training_calls[0]
        assert call["weights_uri"] is None
        assert call["optuna_config"] == OptunaConfig(
            n_startup_trials=1,
            n_trials=1,
            n_jobs=1,
            study_direction=OptimizationDirection.MAXIMIZE,
        )
        assert call["training_config"] == TrainingConfig(
            dynamic_dataset=False,
            train_percentage=1.0,
            threshold_coverage=100,
            load_to_cpu=False,
            epochs=1,
            seed=99,
        )
        assert call["validation_config"] == ValidationConfig(
            validation_mode=CriterionValidation(val_type="loss", batch_size=8)
        )
        # The run's sub-configs are threaded to run_training.
        assert call["broker_config"] == BrokerConfig()
        assert call["connector_config"] == GameServerConnectorConfig()
        assert call["feature_config"] == FeatureConfig()
        # Experiment tags: the whole config as a plain dict (enums kept).
        (tags_args, tags_kwargs) = rec.mlflow.called("set_experiment_tags")[0]
        assert tags_kwargs == {}
        assert tags_args[0] == {
            "optuna_config": {
                "n_startup_trials": 1,
                "n_trials": 1,
                "n_jobs": 1,
                "study_direction": OptimizationDirection.MAXIMIZE,
                "trial_uri": None,
            },
            "training_config": {
                "dynamic_dataset": False,
                "train_percentage": 1.0,
                "threshold_coverage": 100,
                "load_to_cpu": False,
                "epochs": 1,
                "seed": 99,
                "threshold_steps_number": None,
            },
            "validation_config": {
                "validation_mode": {
                    "val_type": "loss",
                    "batch_size": 8,
                    "dataset": TrainingDatasetMode.VALIDATION,
                },
            },
            "mlflow_config": {
                "experiment_name": "CI",
                "tracking_uri": None,
            },
            "broker_config": {
                "port": 35000,
            },
            "game_server_connector_config": {
                "create_connection_timeout_sec": 1,
                "wait_for_socket_reconnection_max_retries": 60,
                "response_timeout_sec": 60,
                "skip_utf_validation": True,
            },
            "feature_config": {
                "verbose_tables": True,
                "disable_message_checks": False,
                "save_if_fail_or_timeout": {
                    "enabled": True,
                    "timeout_sec": 1800,
                    "save_path": REPORT_PATH / "timed_out_agents",
                },
                "on_game_server_restart": {
                    "enabled": True,
                    "wait_for_reset_retries": 600,
                    "wait_for_reset_time": 0.1,
                },
                "dotnet_roll_forward": "LatestPatch",
            },
            "logger_level": logging.INFO,
            "weights_uri": None,
        }

    def test_main_with_tracking_uri(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = default_main_config()
        config["MLFlowConfig"]["tracking_uri"] = "http://127.0.0.1:8080"

        rec = self._main(monkeypatch, tmp_path, config)

        assert rec.mlflow.called("set_tracking_uri") == [
            ((), {"uri": "http://127.0.0.1:8080"})
        ]

    def test_main_forwards_weights_uri(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = default_main_config()
        # The Config validator requires weights_uri and trial_uri together.
        config["weights_uri"] = "mlflow-artifacts:/1/run/artifacts/model.pth"
        config["OptunaConfig"]["trial_uri"] = (
            "mlflow-artifacts:/1/run/artifacts/trial.pkl"
        )

        rec = self._main(monkeypatch, tmp_path, config)

        assert rec.run_training_calls[0]["weights_uri"] == (
            "mlflow-artifacts:/1/run/artifacts/model.pth"
        )

    def test_main_rejects_weights_uri_without_trial_uri(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = default_main_config()
        config["weights_uri"] = "mlflow-artifacts:/1/run/artifacts/model.pth"

        with pytest.raises(
            ValidationError,
            match="can be either None or not None at the same time",
        ):
            self._main(monkeypatch, tmp_path, config)
