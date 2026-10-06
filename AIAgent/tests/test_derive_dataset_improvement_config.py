"""Tests for derive_dataset_improvement_config.

The MLflow fixtures mimic the exact structure ``run_training.py`` produces for
a tuning run: one run per trial (named after the trial number) with
``<epoch>/model.pth`` per epoch, and — after each trial finishes — ``trial.pkl``,
``study.pkl`` and the ``best_trial_number`` tag logged to that same trial's run.
"""

import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import optuna
import pytest
import yaml
from mlflow.tracking import MlflowClient

from derive_dataset_improvement_config import (
    derive_config,
    find_best_trial_run,
    find_experiment,
    find_last_model_epoch,
)
from run_training import BEST_TRIAL_NUMBER_TAG

AIAgent_DIR = Path(__file__).parent.parent
WORKFLOW_CONFIG = AIAgent_DIR.parent / "workflow" / "config_for_tests.yml"


def log_tuning_structure(
    tmp_path: Path, values: list[float], epochs_per_trial: list[int]
):
    """Log a fake tuning run to a file-based MLflow store in ``tmp_path``.

    Mirrors the mlflow calls of ``run_training.py`` (per-trial runs, per-epoch
    model artifacts, and the ``save_study_and_trial`` callback re-entering the
    just finished trial's run). Returns the experiment and a client.
    """
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    experiment = mlflow.set_experiment("CI")
    study = optuna.create_study(direction="minimize")
    for number, (value, epochs) in enumerate(
        zip(values, epochs_per_trial, strict=True)
    ):
        with mlflow.start_run(run_name=str(number)):
            for epoch in range(epochs):
                model = tmp_path / "model.pth"
                model.write_bytes(f"weights-{number}-{epoch}".encode())
                mlflow.log_artifact(str(model), str(epoch))
        trial = study.add_trial(
            optuna.trial.create_trial(
                params={"lr": 0.1 * (number + 1)},
                distributions={"lr": optuna.distributions.FloatDistribution(1e-7, 1.0)},
                values=[value],
            )
        )
        with mlflow.start_run(mlflow.last_active_run().info.run_id):
            trial_pkl = tmp_path / "trial.pkl"
            joblib.dump(trial, trial_pkl)
            mlflow.log_artifact(str(trial_pkl))
            study_pkl = tmp_path / "study.pkl"
            joblib.dump(study, study_pkl)
            mlflow.log_artifact(str(study_pkl))
            mlflow.set_tag(BEST_TRIAL_NUMBER_TAG, study.best_trial.number)
    return experiment, MlflowClient()


def base_config_pointing_at(tmp_path: Path) -> Path:
    """Copy the real workflow config, retargeted at the test's MLflow store."""
    with open(WORKFLOW_CONFIG) as file:
        data = yaml.safe_load(file)
    data["MLFlowConfig"]["tracking_uri"] = f"file://{tmp_path / 'mlruns'}"
    path = tmp_path / "base_config.yml"
    with open(path, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return path


class TestFindBestTrialRun:
    def test_picks_best_trial_not_last_finished(self, tmp_path):
        experiment, client = log_tuning_structure(
            tmp_path, values=[0.5, 0.9], epochs_per_trial=[2, 1]
        )
        best_run = find_best_trial_run(client, experiment.experiment_id)
        assert best_run.info.run_name == "0"

    def test_single_trial(self, tmp_path):
        experiment, client = log_tuning_structure(
            tmp_path, values=[0.9], epochs_per_trial=[1]
        )
        best_run = find_best_trial_run(client, experiment.experiment_id)
        assert best_run.info.run_name == "0"

    def test_no_tagged_runs(self, tmp_path):
        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
        experiment = mlflow.set_experiment("CI")
        client = MlflowClient()
        with pytest.raises(SystemExit, match="best_trial_number"):
            find_best_trial_run(client, experiment.experiment_id)


class TestFindLastModelEpoch:
    def test_highest_epoch_of_the_given_run(self, tmp_path):
        experiment, client = log_tuning_structure(
            tmp_path, values=[0.5, 0.9], epochs_per_trial=[2, 1]
        )
        runs = {
            r.info.run_name: r for r in client.search_runs([experiment.experiment_id])
        }
        assert find_last_model_epoch(client, runs["0"].info.run_id) == 1
        assert find_last_model_epoch(client, runs["1"].info.run_id) == 0

    def test_run_without_model_artifacts(self, tmp_path):
        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
        experiment = mlflow.set_experiment("CI")
        client = MlflowClient()
        with mlflow.start_run(run_name="0"):
            artifact = tmp_path / "other.txt"
            artifact.write_text("x")
            mlflow.log_artifact(str(artifact))
        run = client.search_runs([experiment.experiment_id])[0]
        with pytest.raises(SystemExit, match="model.pth"):
            find_last_model_epoch(client, run.info.run_id)


class TestFindExperiment:
    def test_missing_experiment(self, tmp_path):
        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
        client = MlflowClient()
        with pytest.raises(SystemExit, match="not found"):
            find_experiment(client, "no-such-experiment")


class TestDeriveConfig:
    def test_writes_uris_and_preserves_base_config(self, tmp_path):
        experiment, client = log_tuning_structure(
            tmp_path, values=[0.5, 0.9], epochs_per_trial=[2, 1]
        )
        best_run = next(
            run
            for run in client.search_runs([experiment.experiment_id])
            if run.info.run_name == "0"
        )
        output = tmp_path / "derived" / "config.yml"
        weights_uri, trial_uri = derive_config(
            base_config_pointing_at(tmp_path), output
        )
        assert weights_uri == (
            f"mlflow-artifacts:/{experiment.experiment_id}/{best_run.info.run_id}"
            "/artifacts/1/model.pth"
        )
        assert trial_uri == (
            f"mlflow-artifacts:/{experiment.experiment_id}/{best_run.info.run_id}"
            "/artifacts/trial.pkl"
        )

        with open(output) as file:
            derived = yaml.safe_load(file)
        with open(WORKFLOW_CONFIG) as file:
            base = yaml.safe_load(file)
        assert derived["weights_uri"] == weights_uri
        assert derived["OptunaConfig"]["trial_uri"] == trial_uri
        assert derived["TrainingConfig"] == base["TrainingConfig"]
        assert derived["ValidationConfig"] == base["ValidationConfig"]
        assert {
            key: value
            for key, value in derived["OptunaConfig"].items()
            if key != "trial_uri"
        } == base["OptunaConfig"]

    def test_missing_experiment_fails_without_writing(self, tmp_path):
        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
        base = base_config_pointing_at(tmp_path)
        with open(base) as file:
            data = yaml.safe_load(file)
        data["MLFlowConfig"]["experiment_name"] = "no-such-experiment"
        with open(base, "w") as file:
            yaml.safe_dump(data, file, sort_keys=False)
        output = tmp_path / "derived.yml"
        with pytest.raises(SystemExit, match="not found"):
            derive_config(base, output)
        assert not output.exists()


def test_cli(tmp_path):
    log_tuning_structure(tmp_path, values=[0.5], epochs_per_trial=[1])
    base = base_config_pointing_at(tmp_path)
    output = tmp_path / "derived" / "config.yml"
    result = subprocess.run(
        [
            sys.executable,
            str(AIAgent_DIR / "derive_dataset_improvement_config.py"),
            "--base-config",
            str(base),
            "--output-config",
            str(output),
        ],
        cwd=AIAgent_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with open(output) as file:
        derived = yaml.safe_load(file)
    assert derived["weights_uri"].startswith("mlflow-artifacts:/")
    assert derived["OptunaConfig"]["trial_uri"].endswith("/artifacts/trial.pkl")
