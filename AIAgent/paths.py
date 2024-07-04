from pathlib import Path

REPORT_PATH = Path("./report")
TRAINED_MODELS_PATH = REPORT_PATH / "trained_models"
TRAINING_RESULTS_PATH = REPORT_PATH / "run_tables"
PRETRAINED_MODEL_PATH = REPORT_PATH / "models"
RAW_DATASET_PATH = REPORT_PATH / "SerializedEpisodes"
PROCESSED_DATASET_PATH = REPORT_PATH / "dataset"
OPTUNA_STUDIES_PATH = REPORT_PATH / "optuna_studies"
LOG_PATH = Path("./ml_app.log")
