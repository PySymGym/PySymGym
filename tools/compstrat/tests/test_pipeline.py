from pathlib import Path

import pytest

from compstrat import Args, entrypoint
from src.config_extractor import read_configs

pytestmark = pytest.mark.integration

MOCK_COMPARE_CONFS_DIR = (
    Path(__file__).resolve().parent / "resources" / "mock_compare_confs"
)


@pytest.mark.parametrize("configs_path", sorted(MOCK_COMPARE_CONFS_DIR.glob("*.yaml")))
def test_pipeline_with_mock_data(
    configs_path: Path, mock_runs_dir: Path, tmp_path: Path
):
    args = Args(
        strat1="ALPHA",
        strat2="BETA",
        runs1=[str(mock_runs_dir / "strat_alpha.csv")],
        runs2=[str(mock_runs_dir / "strat_beta.csv")],
        configs=read_configs(configs_path),
        savedir=tmp_path,
    )
    entrypoint(args)
