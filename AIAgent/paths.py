from pathlib import Path

REPORT_PATH = Path("./report")
PRETRAINED_MODEL_PATH = REPORT_PATH / "models"
RAW_DATASET_PATH = REPORT_PATH / "SerializedEpisodes"
PROCESSED_DATASET_PATH = REPORT_PATH / "dataset_path_condition"
LOG_PATH = Path("./ml_app.log")
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
