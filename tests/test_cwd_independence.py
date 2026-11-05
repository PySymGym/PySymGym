"""Tests that nothing depends on the current working directory.

The dataset tools used to append a path relative to the CWD to ``sys.path``
and the integration suites used to resolve resources from the CWD. These tests
pin both properties, plus the entrypoints' artifact paths (``paths.py``) and
the config loading, which used to resolve everything against the CWD.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import paths as paths_module
from common.config.loader import load_config

ROOT = Path(__file__).resolve().parents[1]
DATASET_TOOLS = ["clean", "generate_episodes"]


@pytest.mark.unit
def test_entrypoint_paths_are_anchored_to_the_ai_agent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifact paths in ``paths.py`` do not depend on the CWD at import time."""
    monkeypatch.chdir(tmp_path)
    importlib.reload(paths_module)

    assert paths_module.REPORT_PATH == paths_module.AI_AGENT_PATH / "report"
    assert paths_module.LOG_PATH == paths_module.AI_AGENT_PATH / "ml_app.log"
    assert (
        paths_module.BROKER_LOG_PATH
        == paths_module.AI_AGENT_PATH / "instance_manager.log"
    )
    assert (
        paths_module.trial_dir(1)
        == paths_module.AI_AGENT_PATH / "report" / "trials" / "1"
    )


def _svms_config() -> dict:
    """A production-shaped config (alias keys) with relative path values."""
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
                "PlatformsConfig": [
                    {
                        "name": "dotnet",
                        "DatasetConfigs": [
                            {
                                "dataset_base_path": "maps",
                                "dataset_description": "maps.json",
                            }
                        ],
                        "SVMSInfo": [
                            {
                                "name": "VSharp",
                                "launch_command": "run --port {port}",
                                "min_port": 35000,
                                "max_port": 35050,
                                "server_working_dir": "srv",
                            }
                        ],
                    }
                ],
            }
        },
        "MLFlowConfig": {"experiment_name": "CI"},
    }


@pytest.mark.unit
def test_load_config_anchors_relative_paths_to_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_config resolves relative paths against the config file, not the CWD."""
    # The config lives in a subdir; its relative paths point at siblings.
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "maps").mkdir()
    (cfg_dir / "maps.json").write_text("{}")
    (cfg_dir / "srv").mkdir()
    config_path = cfg_dir / "config.yml"
    config_path.write_text(yaml.safe_dump(_svms_config()))

    monkeypatch.chdir(tmp_path)  # a CWD unrelated to the config
    loaded = load_config(config_path)

    platform = loaded.validation_config.validation_mode.platforms_config[0]
    assert platform.dataset_configs[0].dataset_base_path == (cfg_dir / "maps").resolve()
    assert (
        platform.dataset_configs[0].dataset_description
        == (cfg_dir / "maps.json").resolve()
    )
    assert platform.svms_info[0].server_working_dir == str((cfg_dir / "srv").resolve())


@pytest.mark.unit
def test_load_config_fails_fast_on_missing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path key that does not name an existing object fails the load, naming the key."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "maps.json").write_text("{}")  # only the description exists
    config_path = cfg_dir / "config.yml"
    config_path.write_text(yaml.safe_dump(_svms_config()))

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="config path validation failed") as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    # dataset_base_path and server_working_dir both point at the missing
    # "maps"/"srv" directories; dataset_description is valid.
    assert "dataset_base_path" in message
    assert "server_working_dir" in message
    assert "does not exist" in message
    assert "PlatformsConfig[0].DatasetConfigs[0]" in message
    assert "dataset_description" not in message


@pytest.mark.unit
@pytest.mark.parametrize("module_name", DATASET_TOOLS)
def test_dataset_tool_import_is_side_effect_free(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing a dataset tool must not mutate sys.path or touch the CWD."""
    before = list(sys.path)

    importlib.import_module(module_name)
    assert set(sys.path) - set(before) == set()

    monkeypatch.chdir(tmp_path)
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
@pytest.mark.slow
def test_integration_tests_run_from_a_foreign_cwd(tmp_path: Path) -> None:
    """The coarse integration tier locates its resources from any CWD."""
    targets = [
        ROOT / "AIAgent" / "tests" / "integration" / "test_onnx.py",
        ROOT / "AIAgent" / "tests" / "integration" / "test_statistics.py",
        ROOT / "AIAgent" / "tests" / "integration" / "test_docs_reference.py",
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *(str(target) for target in targets),
            "-q",
            "-m",
            "integration",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
