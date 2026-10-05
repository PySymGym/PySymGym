"""Tests that nothing depends on the current working directory.

The dataset tools used to append a path relative to the CWD to ``sys.path``
and the integration suites used to resolve resources from the CWD. These tests
pin both properties.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET_TOOLS = ["clean", "generate_episodes"]


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
        ROOT / "AIAgent" / "tests" / "test_onnx.py",
        ROOT / "AIAgent" / "tests" / "test_statistics.py",
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
