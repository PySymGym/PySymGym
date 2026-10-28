"""Unit tests for the config path anchoring in ``common.config.loader``."""

import copy
from pathlib import Path

import pytest

from common.config.loader import PATH_KEYS, anchor_paths

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
