"""Self-tests for the shared runstrat fixtures (task #558, Phase 0)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_artifacts_dir_is_temporary(artifacts_dir: Path) -> None:
    assert artifacts_dir.is_dir()
    assert artifacts_dir.name == "artifacts"


def test_for_tests_paths_are_resolved_from_the_tool(
    for_tests_maps_dir: Path, for_tests_maps_description: Path
) -> None:
    from runstrat import DOTNET_VERSION

    assert for_tests_maps_dir.parts[-3:] == (
        "bin",
        "Release",
        f"net{DOTNET_VERSION}",
    )
    assert for_tests_maps_description.name == "for_tests.csv"
