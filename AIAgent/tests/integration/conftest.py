"""Fixtures for the AIAgent integration suite.

Every path here is absolute and resolved from this file, so the suite runs from
any working directory. ``integration/resources`` holds the golden inputs and
configs; ``read_configs`` lists the config files in it and the
``pytest_generate_tests`` hook parametrizes from that list at collection time
so a parametrized test never has to compute a path itself.
"""

import os
from pathlib import Path

import pytest

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def read_configs(dir) -> list[Path]:
    """The ``.yml``/``.yaml`` config files in a directory, as paths."""
    return [
        Path(dir) / file
        for file in os.listdir(Path(dir))
        if file.endswith(".yml") or file.endswith(".yaml")
    ]


@pytest.fixture
def resources_dir() -> Path:
    """The directory holding the integration golden inputs and configs."""
    return RESOURCES_DIR


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the config-driven integration tests from the conftest resources."""
    if "config" in metafunc.fixturenames:
        metafunc.parametrize(
            "config", read_configs(RESOURCES_DIR / "model_configurations")
        )
    if "get_args" in metafunc.fixturenames:
        metafunc.parametrize(
            "get_args",
            read_configs(RESOURCES_DIR / "svms_validation_configs"),
            indirect=True,
        )
