"""Unit tests for the config loading in ``common.config.loader``: path
anchoring, existence validation, and the ``load_config`` seam."""

import copy
from pathlib import Path

import pytest
import yaml

from common.config.config import Config
from common.config.loader import (
    PATH_KEYS,
    anchor_paths,
    load_config,
    validate_paths,
)

pytestmark = pytest.mark.unit


def svms_config() -> dict:
    """A raw svms_each_step config with relative path values."""
    return {
        "OptunaConfig": {"n_trials": 1},
        "ValidationConfig": {
            "validation_mode": {
                "val_type": "svms_each_step",
                "process_count": 1,
                "PlatformsConfig": [
                    {
                        "name": "dotnet",
                        "DatasetConfigs": [
                            {
                                "dataset_base_path": "../maps",
                                "dataset_description": "../workflow/dataset.json",
                            }
                        ],
                        "SVMSInfo": [
                            {
                                "name": "VSharp",
                                "launch_command": "run --port {port}",
                                "server_working_dir": "../GameServers/VSharp",
                                "min_port": 35000,
                                "max_port": 35050,
                            }
                        ],
                    }
                ],
            }
        },
    }


def test_anchor_paths_resolves_relative_values_against_the_anchor(tmp_path: Path):
    anchored = anchor_paths(svms_config(), tmp_path)
    platform = anchored["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0]
    assert platform["DatasetConfigs"][0]["dataset_base_path"] == str(
        (tmp_path / "../maps").resolve()
    )
    assert platform["DatasetConfigs"][0]["dataset_description"] == str(
        (tmp_path / "../workflow/dataset.json").resolve()
    )
    assert platform["SVMSInfo"][0]["server_working_dir"] == str(
        (tmp_path / "../GameServers/VSharp").resolve()
    )


def test_anchor_paths_reaches_paths_inside_a_custom_val_sequence(
    tmp_path: Path,
):
    config = {
        "ValidationConfig": {
            "validation_mode": {
                "val_type": "custom",
                "process_count": 1,
                "val_sequence": [
                    {
                        "val_type": "svms_each_step",
                        "PlatformsConfig": [
                            {
                                "name": "dotnet",
                                "DatasetConfigs": [
                                    {
                                        "dataset_base_path": "maps",
                                        "dataset_description": "maps/dataset.json",
                                    }
                                ],
                                "SVMSInfo": [
                                    {
                                        "name": "VSharp",
                                        "launch_command": "run",
                                        "server_working_dir": "srv",
                                        "min_port": 1,
                                        "max_port": 2,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    }

    anchored = anchor_paths(config, tmp_path)
    dataset = anchored["ValidationConfig"]["validation_mode"]["val_sequence"][0][
        "PlatformsConfig"
    ][0]["DatasetConfigs"][0]
    assert dataset["dataset_base_path"] == str((tmp_path / "maps").resolve())
    assert dataset["dataset_description"] == str(
        (tmp_path / "maps" / "dataset.json").resolve()
    )


def test_anchor_paths_keeps_absolute_values_untouched(tmp_path: Path):
    config = svms_config()
    config["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0][
        "DatasetConfigs"
    ][0]["dataset_base_path"] = "/already/absolute"

    anchored = anchor_paths(config, tmp_path)
    assert (
        anchored["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0][
            "DatasetConfigs"
        ][0]["dataset_base_path"]
        == "/already/absolute"
    )


def test_anchor_paths_does_not_mutate_the_input(tmp_path: Path):
    config = svms_config()
    snapshot = copy.deepcopy(config)

    anchor_paths(config, tmp_path)

    assert config == snapshot


def test_anchor_paths_leaves_non_path_values_untouched(tmp_path: Path):
    config = svms_config()
    config["weights_uri"] = "relative/uri"
    config["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0]["SVMSInfo"][0][
        "launch_command"
    ] = "relative command"

    anchored = anchor_paths(config, tmp_path)
    assert anchored["weights_uri"] == "relative/uri"
    assert (
        anchored["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0][
            "SVMSInfo"
        ][0]["launch_command"]
        == "relative command"
    )


def test_anchor_paths_accepts_path_values(tmp_path: Path):
    config = svms_config()
    config["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0][
        "DatasetConfigs"
    ][0]["dataset_base_path"] = Path("maps")

    anchored = anchor_paths(config, tmp_path)
    assert anchored["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0][
        "DatasetConfigs"
    ][0]["dataset_base_path"] == str((tmp_path / "maps").resolve())


def test_path_keys_table_names_the_three_path_keys():
    assert PATH_KEYS == {
        "dataset_base_path": "directory",
        "dataset_description": "file",
        "server_working_dir": "directory",
    }


def full_config(base: Path) -> dict:
    """A production-shaped config (alias keys) with absolute path values."""
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
            "validation_mode": {
                "val_type": "svms_each_step",
                "process_count": 1,
                "fail_immediately": True,
                "PlatformsConfig": [
                    {
                        "name": "dotnet",
                        "DatasetConfigs": [
                            {
                                "dataset_base_path": str(base / "maps"),
                                "dataset_description": str(base / "maps.json"),
                            }
                        ],
                        "SVMSInfo": [
                            {
                                "name": "VSharp",
                                "launch_command": "run --port {port}",
                                "min_port": 35000,
                                "max_port": 35050,
                                "server_working_dir": str(base / "srv"),
                            }
                        ],
                    }
                ],
            }
        },
        "MLFlowConfig": {"experiment_name": "CI"},
    }


def make_fs(base: Path) -> None:
    """Create the filesystem objects a valid config points at."""
    (base / "maps").mkdir(parents=True)
    (base / "maps.json").write_text("{}")
    (base / "srv").mkdir()


def test_validate_paths_accepts_existing_paths(tmp_path: Path):
    make_fs(tmp_path)

    assert validate_paths(full_config(tmp_path)) == []


def test_validate_paths_names_missing_and_wrong_kind_paths(tmp_path: Path):
    make_fs(tmp_path)
    config = full_config(tmp_path)
    platform = config["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0]
    platform["DatasetConfigs"][0]["dataset_base_path"] = str(tmp_path / "no_such_dir")
    platform["DatasetConfigs"][0]["dataset_description"] = str(tmp_path / "maps")
    platform["SVMSInfo"][0]["server_working_dir"] = str(tmp_path / "maps.json")

    problems = validate_paths(config)

    assert problems == [
        f"dataset_base_path: {tmp_path / 'no_such_dir'} does not exist "
        "(ValidationConfig.validation_mode.PlatformsConfig[0]"
        ".DatasetConfigs[0].dataset_base_path)",
        f"dataset_description: {tmp_path / 'maps'} is not a file "
        "(ValidationConfig.validation_mode.PlatformsConfig[0]"
        ".DatasetConfigs[0].dataset_description)",
        f"server_working_dir: {tmp_path / 'maps.json'} is not a directory "
        "(ValidationConfig.validation_mode.PlatformsConfig[0]"
        ".SVMSInfo[0].server_working_dir)",
    ]


def test_load_config_returns_the_validated_config(tmp_path: Path):
    make_fs(tmp_path)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(full_config(tmp_path)))

    config = load_config(config_path)

    assert isinstance(config, Config)
    platform = config.validation_config.validation_mode.platforms_config[0]
    assert (
        platform.dataset_configs[0].dataset_base_path == (tmp_path / "maps").resolve()
    )
    assert (
        platform.dataset_configs[0].dataset_description
        == (tmp_path / "maps.json").resolve()
    )
    assert platform.svms_info[0].server_working_dir == str((tmp_path / "srv").resolve())


def test_load_config_raises_naming_every_offending_key(tmp_path: Path):
    make_fs(tmp_path)
    config = full_config(tmp_path)
    platform = config["ValidationConfig"]["validation_mode"]["PlatformsConfig"][0]
    platform["DatasetConfigs"][0]["dataset_base_path"] = str(tmp_path / "no_such_dir")
    platform["SVMSInfo"][0]["server_working_dir"] = str(tmp_path / "maps.json")
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="config path validation failed") as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert "dataset_base_path" in message
    assert "does not exist" in message
    assert "server_working_dir" in message
    assert "is not a directory" in message
    assert "PlatformsConfig[0].DatasetConfigs[0]" in message


def test_load_config_propagates_a_missing_config_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no_such_config.yml")
