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


def _epsilon_zeta_args(configs_file: str, mock_runs_dir: Path, tmp_path: Path) -> Args:
    return Args(
        strat1="EPSILON",
        strat2="ZETA",
        runs1=[str(mock_runs_dir / "strat_epsilon.csv")],
        runs2=[str(mock_runs_dir / "strat_zeta.csv")],
        configs=read_configs(MOCK_COMPARE_CONFS_DIR / configs_file),
        savedir=tmp_path,
    )


def test_pipeline_drops_failed_runs_by_default(mock_runs_dir: Path, tmp_path: Path):
    args = _epsilon_zeta_args("compare_confs_1.yaml", mock_runs_dir, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entrypoint(args)

    matplotlib_warnings = _matplotlib_warnings(caught)
    assert not matplotlib_warnings, [str(w.message) for w in matplotlib_warnings]

    # m2 (EPSILON failed) and m3 (ZETA failed) are dropped by default,
    # leaving the inner join {m1}: coverage 100.0 vs 90.0 → EPSILON wins.
    counts = pd.read_csv(tmp_path / "result_count.csv", index_col=0)
    assert list(counts.loc["coverage"]) == [1, 0, 0]


def test_pipeline_keeps_failed_runs_when_disabled(mock_runs_dir: Path, tmp_path: Path):
    args = _epsilon_zeta_args("compare_confs_4.yaml", mock_runs_dir, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entrypoint(args)

    matplotlib_warnings = _matplotlib_warnings(caught)
    assert not matplotlib_warnings, [str(w.message) for w in matplotlib_warnings]

    # drop_failed: false keeps the failed runs: inner join {m1, m2, m3};
    # coverage (higher wins): m1 100>90 EPSILON, m2 -1<80 ZETA,
    # m3 70>-1 EPSILON.
    counts = pd.read_csv(tmp_path / "result_count.csv", index_col=0)
    assert list(counts.loc["coverage"]) == [2, 1, 0]
