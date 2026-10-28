"""CLI seam tests for the compstrat ``main()`` entrypoint.

``main()`` is driven with a patched ``sys.argv`` and a recording fake for
``compstrat.entrypoint`` so no matplotlib comparison run happens.
``read_configs`` runs for real — it is the config-loading seam under test.
"""

import sys
from pathlib import Path

import pytest
import compstrat
from compstrat import Args
from src.config_extractor import read_configs

pytestmark = pytest.mark.unit

FULL_ARGV = [
    "-s1",
    "ALPHA",
    "-r1",
    "a.csv",
    "-s2",
    "BETA",
    "-r2",
    "c.csv",
    "-cp",
    "conf.yaml",
]


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch, mock_compare_confs_dir: Path):
    """Run ``compstrat.main()`` with the given argv; return the captured Args.

    ``-cp`` always points at the first mock compare-configs file so
    ``read_configs`` runs for real.
    """

    def _run(argv: list[str]) -> Args:
        captured: dict = {}

        def fake_entrypoint(args: Args):
            captured["args"] = args

        monkeypatch.setattr(compstrat, "entrypoint", fake_entrypoint)
        argv = list(argv)
        if "-cp" in argv:
            idx = argv.index("-cp")
            argv[idx + 1] = str(mock_compare_confs_dir / "compare_confs_1.yaml")
        monkeypatch.setattr(sys, "argv", ["compstrat"] + argv)
        compstrat.main()
        return captured["args"]

    return _run


def test_main_forwards_parsed_args_to_entrypoint(
    run_main, tmp_path: Path, mock_compare_confs_dir: Path
) -> None:
    savedir = tmp_path / "savedir"

    args = run_main(FULL_ARGV + ["--savedir", str(savedir)])

    assert args == Args(
        strat1="ALPHA",
        runs1=["a.csv"],
        strat2="BETA",
        runs2=["c.csv"],
        savedir=savedir,
        configs=read_configs(mock_compare_confs_dir / "compare_confs_1.yaml"),
    )


def test_main_default_savedir_is_report(run_main) -> None:
    args = run_main(FULL_ARGV)

    assert args.savedir == Path("report")


def test_main_runs_accumulate_across_flags(run_main) -> None:
    args = run_main(
        [
            "-s1",
            "ALPHA",
            "-r1",
            "a.csv",
            "-r1",
            "b.csv",
            "-s2",
            "BETA",
            "-r2",
            "c.csv",
            "-r2",
            "d.csv",
            "-cp",
            "conf.yaml",
        ]
    )

    assert args.runs1 == ["a.csv", "b.csv"]
    assert args.runs2 == ["c.csv", "d.csv"]


@pytest.mark.parametrize("flag", ["-s1", "-r1", "-s2", "-r2", "-cp"])
def test_main_missing_required_arg_exits(run_main, flag: str) -> None:
    argv = []
    skip_next = False
    for token in FULL_ARGV:
        if skip_next:
            skip_next = False
            continue
        if token == flag:
            skip_next = True
            continue
        argv.append(token)

    with pytest.raises(SystemExit):
        run_main(argv)
