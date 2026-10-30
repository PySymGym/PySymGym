"""Unit tests for the compstrat color helper and failure filtering."""

from pathlib import Path

import pandas as pd
import pytest
import compstrat.cli
from compstrat import Args
from compstrat.comparator import (
    Color,
    Comparator,
    CompareConfig,
    DataSourceType,
    Strategy,
)
from compstrat.config_extractor import read_configs

pytestmark = pytest.mark.unit


def test_color_from_hex_normalizes_to_rgba() -> None:
    color = Color.from_hex("#FF7300")
    assert color.to_rgba() == (1.0, 115 / 255, 0.0, 1)


def test_color_keeps_default_alpha() -> None:
    assert Color("orange", 255, 115, 0).to_rgba() == (1.0, 115 / 255, 0.0, 1)


def test_drop_failed_removes_coverage_minus_one() -> None:
    df = pd.DataFrame({"coverage": [1, -1, 2]}, index=["a", "b", "c"])

    result = Comparator._drop_failed(object(), df)

    assert list(result.index) == ["a", "c"]


def test_drop_failed_rebuilds_joins_and_returns_count(tmp_path: Path) -> None:
    comparator = Comparator(
        strat1=_coverage_strategy("A", ["m1", "m2", "m3"], [100.0, -1.0, 3.0]),
        strat2=_coverage_strategy("B", ["m1", "m2", "m4"], [90.0, 80.0, -1.0]),
        savedir=str(tmp_path),
    )

    dropped = comparator.drop_failed()

    assert dropped == 2
    assert list(comparator.strat1.df.index) == ["m1", "m3"]
    assert list(comparator.strat2.df.index) == ["m1", "m2"]
    # The derived state is rebuilt from the filtered dataframes.
    assert list(comparator.inner_df.index) == ["m1"]
    assert list(comparator.outer_df.index) == ["m1", "m2", "m3"]
    assert list(comparator.inner_coverage_eq.index) == []
    means = pd.read_csv(tmp_path / "mean_statistics.csv", index_col=0)
    assert means.loc["coverage_A"].iloc[0] == 100.0
    assert means.loc["coverage_B"].iloc[0] == 90.0


def _run_entrypoint_with_fake_comparator(
    monkeypatch: pytest.MonkeyPatch,
    mock_runs_dir: Path,
    mock_compare_confs_dir: Path,
    configs_file: str,
    tmp_path: Path,
) -> list[str]:
    calls: list[str] = []

    class FakeComparator:
        def __init__(self, **kwargs):
            pass

        def drop_failed(self):
            calls.append("drop_failed")
            return 0

        def compare(self, configs):
            calls.append("compare")

    monkeypatch.setattr(compstrat.cli, "Comparator", FakeComparator)
    args = Args(
        strat1="ALPHA",
        strat2="BETA",
        runs1=[str(mock_runs_dir / "strat_alpha.csv")],
        runs2=[str(mock_runs_dir / "strat_beta.csv")],
        configs=read_configs(mock_compare_confs_dir / configs_file),
        savedir=tmp_path,
    )
    compstrat.cli.entrypoint(args)
    return calls


def test_entrypoint_drops_failed_runs_by_default(
    monkeypatch: pytest.MonkeyPatch,
    mock_runs_dir: Path,
    mock_compare_confs_dir: Path,
    tmp_path: Path,
) -> None:
    # compare_confs_1.yaml sets no drop_failed key: the default (true)
    # applies to every config, so the pipeline drops failed runs.
    calls = _run_entrypoint_with_fake_comparator(
        monkeypatch,
        mock_runs_dir,
        mock_compare_confs_dir,
        "compare_confs_1.yaml",
        tmp_path,
    )

    assert calls == ["drop_failed", "compare"]


def test_entrypoint_keeps_failed_runs_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    mock_runs_dir: Path,
    mock_compare_confs_dir: Path,
    tmp_path: Path,
) -> None:
    # compare_confs_4.yaml sets drop_failed: false: the pipeline-level
    # flag is off for the whole run.
    calls = _run_entrypoint_with_fake_comparator(
        monkeypatch,
        mock_runs_dir,
        mock_compare_confs_dir,
        "compare_confs_4.yaml",
        tmp_path,
    )

    assert calls == ["compare"]


def _coverage_strategy(name: str, index: list[str], coverage: list[float]) -> Strategy:
    color = Color(name, 1, 2, 3)
    return Strategy(
        name=name,
        df=pd.DataFrame(
            {
                "coverage": coverage,
                "coverage_min": coverage,
                "coverage_max": coverage,
            },
            index=pd.Index(index, name="method"),
        ),
        color=color,
        ecolor=color,
    )


def test_outer_join_datasource_covers_union_and_counts_comparable_rows(
    tmp_path: Path,
) -> None:
    comparator = Comparator(
        strat1=_coverage_strategy("A", ["m1", "m2", "m3"], [1.0, 2.0, 3.0]),
        strat2=_coverage_strategy("B", ["m2", "m3", "m4"], [5.0, 3.0, 2.0]),
        savedir=str(tmp_path),
    )

    assert list(comparator.inner_df.index) == ["m2", "m3"]
    assert list(comparator.outer_df.index) == ["m1", "m2", "m3", "m4"]

    # m1 (A only) and m4 (B only) carry NaN on the missing side, so they
    # cannot win, lose, or tie: the outer-join counts equal the
    # hand-computed intersection counts (m2: B wins, m3: tie).
    comparator._compare(
        CompareConfig(
            datasource=DataSourceType.OUTER_JOIN_DF,
            by_column="coverage",
            metric="%",
            exp_name="coverage_outer",
        )
    )
    outer_row = comparator.result_count_df.loc["coverage_outer"]
    assert [outer_row["A_won"], outer_row["B_won"], outer_row["eq"]] == [0, 1, 1]

    comparator._compare(
        CompareConfig(
            datasource=DataSourceType.INNER_JOIN_DF,
            by_column="coverage",
            metric="%",
            exp_name="coverage_inner",
        )
    )
    inner_row = comparator.result_count_df.loc["coverage_inner"]
    assert [inner_row["A_won"], inner_row["B_won"], inner_row["eq"]] == [0, 1, 1]
