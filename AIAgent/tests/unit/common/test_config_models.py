"""Unit tests for the pydantic configuration models in ``common.config``."""

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from common.config.broker_config import BrokerConfig
from common.config.config import Config
from common.config.connector_config import GameServerConnectorConfig
from common.config.dataset_config import DatasetConfig
from common.config.feature_config import (
    FeatureConfig,
    OnGameServerRestartFeature,
    SaveIfFailOrTimeout,
)
from common.config.loader import anchor_paths
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
from common.dataset import TrainingDatasetMode
from common.validation_coverage.svm_info import SVMInfo
from paths import REPORT_PATH

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
        seed=99,
    )


def _validation() -> ValidationConfig:
    return ValidationConfig(
        validation_mode=CriterionValidation(val_type="loss", batch_size=8)
    )


def _mlflow() -> MLFlowConfig:
    return MLFlowConfig(experiment_name="exp")


def _config_kwargs() -> dict:
    return {
        "OptunaConfig": _optuna(),
        "TrainingConfig": _training(),
        "ValidationConfig": _validation(),
        "MLFlowConfig": _mlflow(),
    }


def _config(weights_uri: str | None = None, trial_uri: str | None = None) -> Config:
    kwargs = _config_kwargs()
    kwargs["OptunaConfig"] = _optuna(trial_uri)
    return Config(**kwargs, weights_uri=weights_uri)


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


def test_dataset_config_rejects_relative_paths() -> None:
    # Relative values are ambiguous without an anchor: the config loader
    # resolves them against the config file's location before the model sees
    # them, so the model itself refuses them.
    with pytest.raises(ValidationError, match="not an absolute path"):
        DatasetConfig(
            dataset_base_path="relative/base",
            dataset_description="relative/description.json",
        )


def test_dataset_config_keeps_absolute_paths() -> None:
    config = DatasetConfig(
        dataset_base_path="/abs/base",
        dataset_description="/abs/description.json",
    )
    assert config.dataset_base_path == Path("/abs/base")
    assert config.dataset_description == Path("/abs/description.json")


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
            {"dataset_base_path": "/base", "dataset_description": "/desc.json"}
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
        seed=99,
    )
    assert config.threshold_steps_number is None


def test_broker_config_defaults_match_the_previous_constants() -> None:
    config = BrokerConfig()
    assert config.port == 35000
    assert config.get_ws_url == "http://0.0.0.0:35000/get_ws"
    assert config.post_ws_url == "http://0.0.0.0:35000/post_ws"


def test_broker_config_urls_follow_the_port() -> None:
    config = BrokerConfig(port=1234)
    assert config.get_ws_url == "http://0.0.0.0:1234/get_ws"
    assert config.post_ws_url == "http://0.0.0.0:1234/post_ws"


def test_connector_config_defaults_match_the_previous_constants() -> None:
    config = GameServerConnectorConfig()
    assert config.create_connection_timeout_sec == 1
    assert config.wait_for_socket_reconnection_max_retries == 600
    # An explicit field: no longer derived from the save feature's timeout.
    assert config.response_timeout_sec == 1801
    assert config.skip_utf_validation is True


def test_feature_config_defaults_match_the_previous_constants() -> None:
    config = FeatureConfig()
    assert config.verbose_tables is True
    assert config.disable_message_checks is False
    # Pins the game server's .NET runtime to the major.minor its
    # runtimeconfig.json declares (#669).
    assert config.dotnet_roll_forward == "LatestPatch"
    save = config.save_if_fail_or_timeout
    assert save.enabled is True
    assert save.timeout_sec == 1800
    assert save.save_path == REPORT_PATH / "timed_out_agents"
    restart = config.on_game_server_restart
    assert restart.enabled is True
    assert restart.wait_for_reset_retries == 600
    assert restart.wait_for_reset_time == 0.1


def test_config_exposes_the_merged_sub_configs_with_previous_defaults() -> None:
    config = _config()
    assert config.broker_config.port == 35000
    assert config.game_server_connector_config.response_timeout_sec == 1801
    assert config.feature_config.dotnet_roll_forward == "LatestPatch"
    assert config.logger_level == logging.INFO


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (BrokerConfig, {"port": 1, "unknown": 2}),
        (GameServerConnectorConfig, {"unknown": 1}),
        (FeatureConfig, {"unknown": True}),
        (SaveIfFailOrTimeout, {"unknown": 1}),
        (OnGameServerRestartFeature, {"unknown": 1}),
        (Config, {**_config_kwargs(), "unknown": 1}),
        (
            OptunaConfig,
            {
                "n_startup_trials": 1,
                "n_trials": 2,
                "n_jobs": 1,
                "study_direction": "maximize",
                "unknown": 1,
            },
        ),
        # The stray key that used to sit in the template configs.
        (
            TrainingConfig,
            {
                "dynamic_dataset": False,
                "train_percentage": 0.5,
                "threshold_coverage": 80,
                "load_to_cpu": True,
                "epochs": 10,
                "seed": 99,
                "percentile": 80,
            },
        ),
        (
            ValidationConfig,
            {
                "validation_mode": {"val_type": "loss", "batch_size": 8},
                "unknown": 1,
            },
        ),
        # The stray key that used to sit under a loss mode.
        (
            CriterionValidation,
            {"val_type": "loss", "batch_size": 8, "process_count": 1},
        ),
        (
            SVMValidationSendEachStep,
            {
                "val_type": "svms_each_step",
                "PlatformsConfig": [],
                "process_count": 1,
                "unknown": 1,
            },
        ),
        (
            SVMValidationSendModel,
            {
                "val_type": "svms_model",
                "PlatformsConfig": [],
                "process_count": 1,
                "unknown": 1,
            },
        ),
        (
            CustomValidation,
            {
                "val_type": "custom",
                "val_sequence": [],
                "process_count": 1,
                "unknown": 1,
            },
        ),
        (MLFlowConfig, {"experiment_name": "exp", "unknown": 1}),
        (
            Platform,
            {"name": "p", "DatasetConfigs": [], "SVMSInfo": [], "unknown": 1},
        ),
        (
            DatasetConfig,
            {
                "dataset_base_path": "/base",
                "dataset_description": "/desc.json",
                "unknown": 1,
            },
        ),
        (
            SVMInfo,
            {
                "name": "svm",
                "launch_command": "run",
                "server_working_dir": "/tmp",
                "min_port": 1,
                "max_port": 2,
                "unknown": 1,
            },
        ),
    ],
)
def test_models_reject_unknown_keys(model: type, kwargs: dict) -> None:
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        model(**kwargs)


def test_populate_by_name_accepts_field_names_as_well_as_aliases() -> None:
    # The _config() helper pins the PascalCase alias side of the same fields.
    config = Config(
        optuna_config=_optuna(),
        training_config=_training(),
        validation_config=_validation(),
        mlflow_config=_mlflow(),
    )
    assert config.optuna_config.n_trials == 2

    platform = Platform(name="p", dataset_configs=[], svms_info=[])
    assert platform.dataset_configs == []

    mode = SVMValidationSendEachStep(
        val_type="svms_each_step", platforms_config=[], process_count=1
    )
    assert mode.process_count == 1


REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "template",
    [
        REPO_ROOT / "configs" / "svms_config.yaml",
        REPO_ROOT / "configs" / "val_sequence_config.yaml",
        REPO_ROOT / "configs" / "optuna_config.yaml",
    ],
    ids=lambda path: path.name,
)
def test_shipped_template_configs_construct_cleanly(template: Path) -> None:
    """The shipped template configs match the models (no stray keys).

    The values are anchored (as ``load_config`` does) so the templates'
    relative paths become absolute; existence is not checked, so the pin
    stays hermetic.
    """
    data = yaml.safe_load(template.read_text())
    Config(**anchor_paths(data, template.parent))
