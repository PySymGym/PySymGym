from pathlib import Path
import os

from runstrat import Args, entrypoint, RunMode, DOTNET_VERSION
from src.psstrategy import ExecutionTreeContributedCoverageStrategy

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
PATH_TO_TEST_MAPS = Path(
    f"./resources/ForTests/bin/Release/net{DOTNET_VERSION}"
).resolve()
PATH_TO_TEST_MAPS_DESCRIPTION = Path("./resources/for_tests.csv").resolve()


def test_pipeline_with_mock_data():
    args = Args(
        strategy=ExecutionTreeContributedCoverageStrategy(
            name="ExecutionTreeContributedCoverage"
        ),
        timeout=100,
        pysymgym_path=Path("../..").resolve(),
        savedir=ARTIFACTS_DIR,
        assembly_infos=[(PATH_TO_TEST_MAPS, PATH_TO_TEST_MAPS_DESCRIPTION)],
        run_mode=RunMode.DEBUG,
    )
    entrypoint(args)
