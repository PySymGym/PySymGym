from pathlib import Path

import pytest

from runstrat import Args, entrypoint, RunMode
from src.psstrategy import ExecutionTreeContributedCoverageStrategy

pytestmark = pytest.mark.e2e

PYSYMGYM_ROOT = Path(__file__).resolve().parents[3]


def test_pipeline_with_mock_data(
    artifacts_dir: Path,
    for_tests_maps_dir: Path,
    for_tests_maps_description: Path,
):
    args = Args(
        strategy=ExecutionTreeContributedCoverageStrategy(
            name="ExecutionTreeContributedCoverage"
        ),
        timeout=100,
        pysymgym_path=PYSYMGYM_ROOT,
        savedir=artifacts_dir,
        assembly_infos=[(for_tests_maps_dir, for_tests_maps_description)],
        run_mode=RunMode.DEBUG,
    )
    entrypoint(args)
