"""Unit tests for the compstrat comparison config extraction."""

from pathlib import Path

import pytest
from src.comparator import CompareConfig, DataSourceType
from src.config_extractor import _structure_config, read_configs

pytestmark = pytest.mark.unit


def test_read_configs_parses_checked_in_yaml(mock_compare_confs_dir: Path) -> None:
    configs = read_configs(mock_compare_confs_dir / "compare_confs_1.yaml")

    assert configs
    assert all(isinstance(config, CompareConfig) for config in configs)
    assert configs[0].datasource is DataSourceType.INNER_JOIN_DF
    assert configs[0].exp_name == "coverage"


def test_structure_config_maps_enum_by_value() -> None:
    config = _structure_config(
        {"datasource": "OUTER_JOIN_DF", "by_column": "coverage", "metric": "%"}
    )

    assert config.datasource is DataSourceType.OUTER_JOIN_DF
    assert config.divider_line is False
