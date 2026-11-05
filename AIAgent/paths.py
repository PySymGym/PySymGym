from pathlib import Path

AI_AGENT_PATH = Path(__file__).resolve().parent
REPO_ROOT = AI_AGENT_PATH.parent
RESOURCES_PATH = REPO_ROOT / "resources"

# Artifact root, anchored to the AIAgent directory (located via __file__, the
# same mechanism RESOURCES_PATH uses for the repo root) instead of the CWD:
# running the entrypoints from anywhere writes the generated artifacts
# (models, trials, datasets, study/trial pickles, the app log) to the same
# place. Everything under it is generated at runtime and git-ignored.
REPORT_PATH = AI_AGENT_PATH / "report"
PRETRAINED_MODEL_PATH = REPORT_PATH / "models"
RAW_DATASET_PATH = REPORT_PATH / "SerializedEpisodes"
PROCESSED_DATASET_PATH = REPORT_PATH / "dataset_path_condition"
LOG_PATH = AI_AGENT_PATH / "ml_app.log"
BROKER_LOG_PATH = AI_AGENT_PATH / "instance_manager.log"
MODEL_FILE_NAME = "model.pth"
CURRENT_STUDY_PATH = REPORT_PATH / "study.pkl"
CURRENT_TRIAL_PATH = REPORT_PATH / "trial.pkl"
TRIALS_PATH = REPORT_PATH / "trials"


def trial_dir(trial_number: int) -> Path:
    """Artifact directory of one Optuna trial.

    Parameters
    ----------
    trial_number : int
        The Optuna trial number.

    Returns
    -------
    Path
        ``report/trials/<trial_number>/``.
    """
    return TRIALS_PATH / str(trial_number)
