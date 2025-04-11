from pathlib import Path
import os

from runstrat import Args, entrypoint, RunMode
from src.psstrategy import BasePSStrategy

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def test_pipeline_with_mock_data():
    args = Args(
        strategy=BasePSStrategy(name="ExecutionTreeContributedCoverage"),
        timeout=100,
        pysymgym_path=Path("../..").absolute(),
        savedir=ARTIFACTS_DIR,
        assembly_infos=[
            (
                Path("./resources/ForTests/bin/Release/net7.0").absolute(),
                Path("./resources/for_tests.csv").absolute(),
            )
        ],
        run_mode=RunMode.DEBUG,
    )
    entrypoint(args)
