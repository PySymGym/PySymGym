import os
import pathlib
import tempfile
import warnings

import pytest

from compstrat import Args, entrypoint
from src.config_extractor import read_configs

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
            runs1=["tests/resources/mock_runs/strat_alpha.csv"],
            runs2=["tests/resources/mock_runs/strat_beta.csv"],
            configs=read_configs(configs_path),
            savedir=tmpdirname,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entrypoint(args)

    # The mock data contains single-row groups, which make matplotlib call
    # float() on length-1 Series/ndarrays and emit a FutureWarning or
    # DeprecationWarning (#617). Guard against that regression.
    matplotlib_warnings = [w for w in caught if "matplotlib" in (w.filename or "")]
    assert not matplotlib_warnings, [str(w.message) for w in matplotlib_warnings]
