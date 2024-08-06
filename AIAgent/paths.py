from pathlib import Path

REPORT_PATH = Path("./report")
PRETRAINED_MODEL_PATH = REPORT_PATH / "models"
RAW_DATASET_PATH = REPORT_PATH / "SerializedEpisodes"
PROCESSED_DATASET_PATH = REPORT_PATH / "dataset"
LOG_PATH = Path("./ml_app.log")
CURRENT_MODEL_PATH = REPORT_PATH / "model.pth"
CURRENT_STUDY_PATH = REPORT_PATH / "study.pkl"
CURRENT_TABLE_PATH = REPORT_PATH / "svms_result_table.log"
