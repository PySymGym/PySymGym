"""Smoke tests for the shared pytest infrastructure (task #558, Phase 0).

These guard the root configuration every component suite depends on: the marker
taxonomy is registered, the import mode is ``importlib`` (required so the three
same-named ``tests/test_pipeline.py`` modules can coexist), and every component
has a test directory wired into ``testpaths``.
"""

from pathlib import Path
from typing import cast

import pytest
from pytest import Config

pytestmark = pytest.mark.unit

UNIT_MARKERS = {
    "unit",
    "integration",
    "e2e",
    "gpu",
    "network",
    "slow",
    "serial",
}

COMPONENT_TEST_DIRS = [
    "AIAgent/tests",
    "tools/compstrat/tests",
    "tools/runstrat/tests",
    "tools/dataset_tools/tests",
    "tools/util/tests",
]

ROOT = Path(__file__).resolve().parent.parent


def test_marker_taxonomy_is_registered(pytestconfig: Config) -> None:
    markers = cast("list[str]", pytestconfig.getini("markers"))
    registered = {line.split(":", 1)[0].strip() for line in markers if line.strip()}
    assert UNIT_MARKERS <= registered


def test_import_mode_is_importlib(pytestconfig: Config) -> None:
    assert pytestconfig.getoption("importmode") == "importlib"


def test_every_component_test_dir_is_on_testpaths(pytestconfig: Config) -> None:
    configured = cast("list[str]", pytestconfig.getini("testpaths"))
    testpaths = {str(Path(path)) for path in configured}
    for component_dir in COMPONENT_TEST_DIRS:
        assert component_dir in testpaths


def test_every_component_contributes_at_least_one_test() -> None:
    for component_dir in COMPONENT_TEST_DIRS:
        directory = ROOT / component_dir
        test_files = list(directory.rglob("test_*.py"))
        assert test_files, f"{component_dir} has no test files"
