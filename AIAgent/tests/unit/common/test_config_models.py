"""Unit tests for the pydantic configuration models in ``common.config``."""

from pathlib import Path

import pytest
from ml.dataset import TrainingDatasetMode
from pydantic import ValidationError

from common.config.config import Config
from common.config.dataset_config import DatasetConfig
from common.config.mlflow_config import MLFlowConfig
from common.config.optuna_config import OptimizationDirection, OptunaConfig
from common.config.platform_config import Platform
from common.config.training_config import TrainingConfig
from common.config.validation_config import (
    CriterionValidation,
    CustomValidation,
    SVMValidationSendEachStep,
    SVMValidationSendModel,
    ValidationConfig,
)
from common.validation_coverage.svm_info import SVMInfo

pytestmark = pytest.mark.unit


def _optuna(trial_uri: str | None = None) -> OptunaConfig:
    return OptunaConfig(
        n_startup_trials=1,
        n_trials=2,
        n_jobs=1,
        study_direction=OptimizationDirection.MAXIMIZE,
        trial_uri=trial_uri,
    )


def _training() -> TrainingConfig:
    return TrainingConfig(
        dynamic_dataset=False,
        train_percentage=0.5,
        threshold_coverage=80,
        load_to_cpu=True,
        epochs=10,
    )


def _validation() -> ValidationConfig:
    return ValidationConfig(
        validation_mode=CriterionValidation(val_type="loss", batch_size=8)
    )


def _mlflow() -> MLFlowConfig:
    return MLFlowConfig(experiment_name="exp")


def _config(weights_uri: str | None = None, trial_uri: str | None = None) -> Config:
    return Config(
        OptunaConfig=_optuna(trial_uri),
        TrainingConfig=_training(),
        ValidationConfig=_validation(),
        MLFlowConfig=_mlflow(),
        weights_uri=weights_uri,
    )


@pytest.mark.parametrize(
    ("weights_uri", "trial_uri"),
    [(None, None), ("weights", "study"), (None, "study"), ("weights", None)],
)
def test_config_requires_both_or_neither_uri(
    weights_uri: str | None, trial_uri: str | None
) -> None:
    if (weights_uri is None) == (trial_uri is None):
        assert _config(weights_uri, trial_uri).weights_uri == weights_uri
    else:
        with pytest.raises(
            ValidationError, match="either None or not None at the same time"
        ):
            _config(weights_uri, trial_uri)


def test_validation_config_parses_criterion_mode() -> None:
    config = ValidationConfig(validation_mode={"val_type": "loss", "batch_size": 8})
    assert isinstance(config.validation_mode, CriterionValidation)
    assert config.validation_mode.dataset is TrainingDatasetMode.VALIDATION


def test_validation_config_parses_svms_each_step_mode() -> None:
    config = ValidationConfig(
        validation_mode={
            "val_type": "svms_each_step",
            "PlatformsConfig": [],
            "process_count": 2,
        }
    )
    assert isinstance(config.validation_mode, SVMValidationSendEachStep)
    assert config.validation_mode.process_count == 2


def test_validation_config_parses_svms_model_mode() -> None:
    config = ValidationConfig(
        validation_mode={
            "val_type": "svms_model",
            "PlatformsConfig": [],
            "process_count": 4,
            "fail_immediately": True,
        }
    )
    assert isinstance(config.validation_mode, SVMValidationSendModel)
    assert config.validation_mode.fail_immediately is True


def test_validation_config_parses_custom_mode() -> None:
    config = ValidationConfig(
        validation_mode={
            "val_type": "custom",
            "val_sequence": [{"val_type": "loss", "batch_size": 4}],
            "process_count": 1,
        }
    )
    assert isinstance(config.validation_mode, CustomValidation)
    assert isinstance(config.validation_mode.val_sequence[0], CriterionValidation)


def test_validation_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(validation_mode={"val_type": "unknown"})


def test_dataset_config_resolves_paths_against_cwd() -> None:
    config = DatasetConfig(
        dataset_base_path="relative/base",
        dataset_description="relative/description.json",
    )
    assert config.dataset_base_path == Path("relative/base").resolve()
    assert config.dataset_description == Path("relative/description.json").resolve()


def test_svm_info_to_dict_exposes_all_fields() -> None:
    info = SVMInfo(
        name="svm",
        launch_command="run",
        server_working_dir="/tmp",
        min_port=1,
        max_port=2,
    )
    assert info.to_dict() == {
        "name": "svm",
        "launch_command": "run",
        "server_working_dir": "/tmp",
        "min_port": 1,
        "max_port": 2,
    }


def test_platform_parses_dataset_and_svm_lists() -> None:
    platform = Platform(
        name="platform",
        DatasetConfigs=[
            {"dataset_base_path": "base", "dataset_description": "desc.json"}
        ],
        SVMSInfo=[
            {
                "name": "svm",
                "launch_command": "run",
                "server_working_dir": "/tmp",
                "min_port": 1,
                "max_port": 2,
            }
        ],
    )
    assert platform.name == "platform"
    assert isinstance(platform.dataset_configs[0], DatasetConfig)
    assert isinstance(platform.svms_info[0], SVMInfo)


def test_optuna_config_stores_direction_and_default_uri() -> None:
    config = OptunaConfig(
        n_startup_trials=5,
        n_trials=10,
        n_jobs=2,
        study_direction=OptimizationDirection.MINIMIZE,
    )
    assert config.study_direction is OptimizationDirection.MINIMIZE
    assert config.trial_uri is None


def test_mlflow_config_tracking_uri_defaults_to_none() -> None:
    assert MLFlowConfig(experiment_name="exp").tracking_uri is None


def test_training_config_threshold_steps_defaults_to_none() -> None:
    config = TrainingConfig(
        dynamic_dataset=True,
        train_percentage=0.7,
        threshold_coverage=90,
        load_to_cpu=False,
        epochs=3,
    )
    assert config.threshold_steps_number is None
