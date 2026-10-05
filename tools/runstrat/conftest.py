"""Shared fixtures for the runstrat test suite.

Paths are resolved relative to this file so the suite collects and runs from
any working directory; artifacts are written under ``tmp_path`` instead of a
repository-relative directory.
"""

from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = TOOL_DIR / "resources"


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    """A temporary directory for runstrat run artifacts."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return artifacts


@pytest.fixture
def for_tests_maps_dir() -> Path:
    """The built ``ForTests`` .NET assembly directory used by the e2e test."""
    from runstrat import DOTNET_VERSION

    return RESOURCES_DIR / "ForTests" / "bin" / "Release" / f"net{DOTNET_VERSION}"


@pytest.fixture
def for_tests_maps_description() -> Path:
    """The description CSV for the built ``ForTests`` maps."""
    return RESOURCES_DIR / "for_tests.csv"
