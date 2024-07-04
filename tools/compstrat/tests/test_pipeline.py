import os
import pathlib
import tempfile

import pytest

from compstrat import Args, entrypoint
from src.compstrat_config_extractor import read_configs

MOCK_CONFIGS_DIR = pathlib.Path("tests/resources/mock_compare_confs")


@pytest.mark.parametrize(
    "configs_path",
    [
        pathlib.Path(os.path.join(MOCK_CONFIGS_DIR, f))
        for f in os.listdir(MOCK_CONFIGS_DIR)
        if os.path.isfile(os.path.join(MOCK_CONFIGS_DIR, f))
    ],
)
def test_pipeline_with_mock_data(configs_path: pathlib.Path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        args = Args(
            strat1="ALPHA",
            strat2="BETA",
            run1="tests/resources/mock_runs/strat_alpha.csv",
            run2="tests/resources/mock_runs/strat_beta.csv",
            configs=read_configs(configs_path),
            savedir=tmpdirname,
        )
        entrypoint(args)
