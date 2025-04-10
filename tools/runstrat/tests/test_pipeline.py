from pathlib import Path
import tempfile

from runstrat import Args, entrypoint
from src.psstrategy import BasePSStrategy


def test_pipeline_with_mock_data():
    # with tempfile.TemporaryDirectory() as tmpdirname:
    args = Args(
        strategy=BasePSStrategy(name="ExecutionTreeContributedCoverage"),
        timeout=100,
        pysymgym_path=Path("../..").absolute(),
        savedir="./",
        assembly_infos=[
            (
                Path("./resources/ForTests/bin/Release/net7.0").absolute(),
                Path("./resources/for_tests.csv").absolute(),
            )
        ],
    )
    entrypoint(args)
