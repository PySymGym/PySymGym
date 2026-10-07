"""Derive a dataset-improvement config from a tuning run's MLflow artifacts.

The main loop alternates hyper-parameter tuning and dataset improvement, and
the improvement run must start from the tuning run's best model: its final
weights (``model.pth``) and the frozen trial that produced them (``trial.pkl``),
per the contract in ``docs/usage.rst``. This tool queries the MLflow server
named by the base config's ``MLFlowConfig`` section, locates those artifacts,
and writes the base config plus ``weights_uri`` and ``OptunaConfig.trial_uri``.
"""

import argparse
import re
from pathlib import Path

import mlflow
import yaml
from common.config.config import Config
from mlflow.tracking import MlflowClient
from paths import CURRENT_MODEL_PATH, CURRENT_TRIAL_PATH
from run_training import BEST_TRIAL_NUMBER_TAG


def find_experiment(client: MlflowClient, name: str):
    """Return the MLflow experiment with the given name.

    Parameters
    ----------
    client : MlflowClient
        Client connected to the tracking server to query.
    name : str
        Experiment name.

    Returns
    -------
    mlflow.entities.Experiment
        The experiment.

    Raises
    ------
    SystemExit
        If no such experiment exists.
    """
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        raise SystemExit(f"MLflow experiment {name!r} not found")
    return experiment


def find_best_trial_run(client: MlflowClient, experiment_id: str):
    """Return the MLflow run of the tuning's best trial.

    Tuning (``run_training.py``) runs each trial in its own MLflow run named
    after the trial number and, after each trial finishes, logs a
    ``best_trial_number`` tag holding the best trial among those completed so
    far. The overall best is therefore the tag value on the most recently
    finished tagged run; the best trial's own run is the run named after that
    number from the same tuning session.

    Parameters
    ----------
    client : MlflowClient
        Client connected to the tracking server to query.
    experiment_id : str
        Experiment holding the tuning runs.

    Returns
    -------
    mlflow.entities.Run
        The best trial's run.

    Raises
    ------
    SystemExit
        If no tuning run is found in the experiment.
    """
    runs = client.search_runs([experiment_id])
    tagged = [
        run
        for run in runs
        if BEST_TRIAL_NUMBER_TAG in run.data.tags and run.info.end_time is not None
    ]
    if not tagged:
        raise SystemExit(
            f"no finished run in experiment {experiment_id} carries the "
            f"{BEST_TRIAL_NUMBER_TAG!r} tag; did the tuning run complete?"
        )
    last = max(tagged, key=lambda run: run.info.end_time)
    best_number = int(last.data.tags[BEST_TRIAL_NUMBER_TAG])
    candidates = [
        run
        for run in runs
        if run.info.run_name == str(best_number)
        and run.info.end_time is not None
        and run.info.end_time <= last.info.end_time
    ]
    if not candidates:
        raise SystemExit(
            f"no run named {best_number!r} found in experiment {experiment_id}"
        )
    return max(candidates, key=lambda run: run.info.end_time)


def _artifact_paths(
    client: MlflowClient, run_id: str, path: str | None = None
) -> list[str]:
    """Recursively list a run's artifact paths (the client lists one level)."""
    paths = []
    for artifact in client.list_artifacts(run_id, path=path):
        if artifact.is_dir:
            paths.extend(_artifact_paths(client, run_id, artifact.path))
        else:
            paths.append(artifact.path)
    return paths


def find_last_model_epoch(client: MlflowClient, run_id: str) -> int:
    """Return the highest epoch whose ``<epoch>/model.pth`` was logged to a run.

    Parameters
    ----------
    client : MlflowClient
        Client connected to the tracking server to query.
    run_id : str
        Run to inspect.

    Returns
    -------
    int
        The last (highest) epoch with a logged model artifact.

    Raises
    ------
    SystemExit
        If the run has no ``<epoch>/model.pth`` artifacts.
    """
    pattern = re.compile(rf"(\d+)/{re.escape(CURRENT_MODEL_PATH.name)}")
    epochs = [
        int(match.group(1))
        for path in _artifact_paths(client, run_id)
        if (match := pattern.fullmatch(path))
    ]
    if not epochs:
        raise SystemExit(
            f"no {CURRENT_MODEL_PATH.name!r} artifact found in run {run_id}"
        )
    return max(epochs)


def derive_config(base_config_path: Path, output_config_path: Path) -> tuple[str, str]:
    """Derive the dataset-improvement config from the tuning run's artifacts.

    Parameters
    ----------
    base_config_path : Path
        Dataset-improvement config template; its ``MLFlowConfig`` section names
        the tracking server and experiment to query.
    output_config_path : Path
        Where to write the base config plus the derived URIs.

    Returns
    -------
    tuple[str, str]
        The derived ``(weights_uri, trial_uri)`` pair.

    Raises
    ------
    SystemExit
        If the tuning artifacts cannot be located or the derived config fails
        validation; nothing is written in that case.
    """
    with open(base_config_path, "r") as file:
        data = yaml.safe_load(file)
    mlflow_section = data.get("MLFlowConfig", {})
    tracking_uri = mlflow_section.get("tracking_uri")
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_section.get("experiment_name")
    if experiment_name is None:
        raise SystemExit(f"{base_config_path} has no MLFlowConfig.experiment_name")

    client = MlflowClient()
    experiment = find_experiment(client, experiment_name)
    best_run = find_best_trial_run(client, experiment.experiment_id)
    epoch = find_last_model_epoch(client, best_run.info.run_id)
    weights_uri = (
        f"mlflow-artifacts:/{experiment.experiment_id}/{best_run.info.run_id}"
        f"/artifacts/{epoch}/{CURRENT_MODEL_PATH.name}"
    )
    trial_uri = (
        f"mlflow-artifacts:/{experiment.experiment_id}/{best_run.info.run_id}"
        f"/artifacts/{CURRENT_TRIAL_PATH.name}"
    )

    if "OptunaConfig" not in data:
        raise SystemExit(f"{base_config_path} has no OptunaConfig section")
    data["weights_uri"] = weights_uri
    data["OptunaConfig"]["trial_uri"] = trial_uri
    # Reuse the config model's validation (notably that weights_uri and
    # trial_uri must be set together) before writing anything.
    Config(**data)

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    print(f"weights_uri: {weights_uri}")
    print(f"trial_uri: {trial_uri}")
    print(f"derived config: {output_config_path}")
    return weights_uri, trial_uri


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a dataset-improvement config from a tuning run's MLflow "
            "artifacts (best trial's model.pth and trial.pkl)."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Dataset-improvement config template to extend with the URIs.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        required=True,
        help="Where to write the derived config.",
    )
    args = parser.parse_args()
    derive_config(args.base_config, args.output_config)


if __name__ == "__main__":
    main()
