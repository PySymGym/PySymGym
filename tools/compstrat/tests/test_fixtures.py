"""Self-tests for the shared compstrat fixtures (task #558, Phase 0)."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.unit
def test_mock_run_dataframe_is_loaded(mock_run_dataframe: pd.DataFrame) -> None:
    assert not mock_run_dataframe.empty
    assert "method" in mock_run_dataframe.columns


@pytest.mark.unit
def test_mock_compare_confs_dir_contains_configs(mock_compare_confs_dir: Path) -> None:
    assert mock_compare_confs_dir.is_dir()
    assert list(mock_compare_confs_dir.glob("*.yaml"))
