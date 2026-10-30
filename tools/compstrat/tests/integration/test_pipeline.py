import warnings
from pathlib import Path

import pandas as pd
import pytest

from compstrat import Args, entrypoint
from compstrat.config_extractor import read_configs

pytestmark = pytest.mark.integration

MOCK_COMPARE_CONFS_DIR = (
    Path(__file__).resolve().parent.parent / "resources" / "mock_compare_confs"
)


def _matplotlib_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [w for w in caught if "matplotlib" in (w.filename or "")]


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entrypoint(args)

    # The mock data contains single-row groups, which make matplotlib call
    # float() on length-1 Series/ndarrays and emit a FutureWarning or
    # DeprecationWarning (#617). Guard against that regression.
    matplotlib_warnings = _matplotlib_warnings(caught)
    assert not matplotlib_warnings, [str(w.message) for w in matplotlib_warnings]


def test_pipeline_outer_join_with_disjoint_methods(mock_runs_dir: Path, tmp_path: Path):
    args = Args(
        strat1="GAMMA",
        strat2="DELTA",
        runs1=[str(mock_runs_dir / "strat_gamma.csv")],
        runs2=[str(mock_runs_dir / "strat_delta.csv")],
        configs=read_configs(MOCK_COMPARE_CONFS_DIR / "compare_confs_3.yaml"),
        savedir=tmp_path,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entrypoint(args)

    matplotlib_warnings = _matplotlib_warnings(caught)
    assert not matplotlib_warnings, [str(w.message) for w in matplotlib_warnings]

    # m1 (GAMMA only) and m6 (DELTA only) have no counterpart on the other
    # side and are excluded from every count; the intersection is m2-m5.
    counts = pd.read_csv(tmp_path / "result_count.csv", index_col=0)
    assert list(counts.loc["coverage"]) == [0, 3, 1]
    assert list(counts.loc["tests"]) == [0, 4, 0]
    assert list(counts.loc["errors"]) == [4, 0, 0]
