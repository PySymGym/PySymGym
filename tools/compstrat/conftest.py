"""Shared fixtures for the compstrat test suite.

Paths are resolved relative to this file so the suite collects and runs from
any working directory, including the repository root.
"""

from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent / "tests"
RESOURCES_DIR = TESTS_DIR / "resources"


@pytest.fixture
def mock_compare_confs_dir() -> Path:
    """Directory holding the mock comparator configurations."""
    return RESOURCES_DIR / "mock_compare_confs"


@pytest.fixture
def mock_runs_dir() -> Path:
    """Directory holding the mock strategy run CSV files."""
    return RESOURCES_DIR / "mock_runs"


@pytest.fixture
def mock_run_dataframe(mock_runs_dir: Path) -> pd.DataFrame:
    """The mock strategy run CSV loaded as a DataFrame."""
    return pd.read_csv(mock_runs_dir / "strat_alpha.csv")
