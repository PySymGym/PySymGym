"""CLI seam tests for the runstrat ``main()`` entrypoint.

``main()`` is driven with a patched ``sys.argv`` and a recording fake for
``runstrat.cli.entrypoint`` so no end-to-end strategy run happens.
``os.makedirs`` is faked too, so the default (timestamp-named) savedir never
touches the real filesystem. ``BasePSStrategy.parse`` runs for real — it is
the dispatch under test.
"""

import os
import sys
from pathlib import Path

import pytest
import runstrat.cli
from runstrat import RunMode
from runstrat.psstrategy import AIStrategy, ExecutionTreeContributedCoverageStrategy

pytestmark = pytest.mark.unit


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch):
    """Run ``runstrat.main()`` with the given argv; return the captured state."""

    def _run(argv: list[str]):
        captured: dict = {}
        makedirs_calls: list = []

        def fake_entrypoint(args):
            captured["args"] = args
            return "SENTINEL"

        def fake_makedirs(path, *args, **kwargs):
            makedirs_calls.append(path)

        monkeypatch.setattr(runstrat.cli, "entrypoint", fake_entrypoint)
        monkeypatch.setattr(os, "makedirs", fake_makedirs)
        monkeypatch.setattr(sys, "argv", ["runstrat"] + argv)
        result = runstrat.cli.main()
        return captured, makedirs_calls, result

    return _run


def test_main_forwards_parsed_args_to_entrypoint(run_main, tmp_path: Path) -> None:
    savedir = tmp_path / "savedir"
    pysymgym = tmp_path / "pysymgym"
    dlls = tmp_path / "dlls"
    csv = tmp_path / "maps.csv"

    captured, makedirs_calls, _ = run_main(
        [
            "-s",
            "ExecutionTreeContributedCoverage",
            "-t",
            "5",
            "-ps",
            str(pysymgym),
            "-sd",
            str(savedir),
            "-as",
            str(dlls),
            str(csv),
        ]
    )

    args = captured["args"]
    assert isinstance(args.strategy, ExecutionTreeContributedCoverageStrategy)
    assert args.timeout == 5
    assert args.pysymgym_path == pysymgym
    assert args.savedir == savedir
    # ``-as`` with action=append + nargs=2 accumulates a list of 2-lists.
    assert args.assembly_infos == [[dlls, csv]]
    assert args.run_mode is RunMode.BENCHMARKS
    assert makedirs_calls == [savedir]


def test_main_default_savedir_is_the_timestamp(run_main, tmp_path: Path) -> None:
    pysymgym = tmp_path / "pysymgym"
    dlls = tmp_path / "dlls"
    csv = tmp_path / "maps.csv"

    captured, makedirs_calls, _ = run_main(
        [
            "-s",
            "ExecutionTreeContributedCoverage",
            "-t",
            "5",
            "-ps",
            str(pysymgym),
            "-as",
            str(dlls),
            str(csv),
        ]
    )

    expected = Path(f"{runstrat.cli.timestamp}")
    assert captured["args"].savedir == expected
    assert makedirs_calls == [expected]


def test_main_multiple_assembly_infos_accumulate(run_main, tmp_path: Path) -> None:
    pysymgym = tmp_path / "pysymgym"
    d1, l1 = tmp_path / "d1", tmp_path / "l1"
    d2, l2 = tmp_path / "d2", tmp_path / "l2"

    captured, _, _ = run_main(
        [
            "-s",
            "ExecutionTreeContributedCoverage",
            "-t",
            "5",
            "-ps",
            str(pysymgym),
            "-sd",
            str(tmp_path / "savedir"),
            "-as",
            str(d1),
            str(l1),
            "-as",
            str(d2),
            str(l2),
        ]
    )

    assert captured["args"].assembly_infos == [[d1, l1], [d2, l2]]


def test_main_ai_strategy_carries_the_model_path(run_main, tmp_path: Path) -> None:
    pysymgym = tmp_path / "pysymgym"
    model = tmp_path / "model.onnx"
    dlls = tmp_path / "dlls"
    csv = tmp_path / "maps.csv"

    captured, _, _ = run_main(
        [
            "-s",
            "AI",
            "-mp",
            str(model),
            "-t",
            "5",
            "-ps",
            str(pysymgym),
            "-sd",
            str(tmp_path / "savedir"),
            "-as",
            str(dlls),
            str(csv),
        ]
    )

    strategy = captured["args"].strategy
    assert isinstance(strategy, AIStrategy)
    assert strategy.model_path == model


def test_main_missing_strategy_exits(run_main) -> None:
    with pytest.raises(SystemExit):
        run_main(["-t", "5"])


def test_main_missing_timeout_exits(run_main) -> None:
    with pytest.raises(SystemExit):
        run_main(["-s", "ExecutionTreeContributedCoverage"])


def test_main_unknown_strategy_raises_value_error(run_main, tmp_path: Path) -> None:
    pysymgym = tmp_path / "pysymgym"
    dlls = tmp_path / "dlls"
    csv = tmp_path / "maps.csv"

    with pytest.raises(ValueError, match="Unknown strategy"):
        run_main(
            [
                "-s",
                "bogus",
                "-t",
                "5",
                "-ps",
                str(pysymgym),
                "-sd",
                str(tmp_path / "savedir"),
                "-as",
                str(dlls),
                str(csv),
            ]
        )


def test_main_ai_without_model_path_asserts(run_main, tmp_path: Path) -> None:
    pysymgym = tmp_path / "pysymgym"
    dlls = tmp_path / "dlls"
    csv = tmp_path / "maps.csv"

    with pytest.raises(AssertionError):
        run_main(
            [
                "-s",
                "AI",
                "-t",
                "5",
                "-ps",
                str(pysymgym),
                "-sd",
                str(tmp_path / "savedir"),
                "-as",
                str(dlls),
                str(csv),
            ]
        )


def test_main_missing_pysymgym_path_is_an_argparse_error(
    run_main, tmp_path: Path
) -> None:
    dlls = tmp_path / "dlls"
    csv = tmp_path / "maps.csv"

    with pytest.raises(SystemExit):
        run_main(
            [
                "-s",
                "ExecutionTreeContributedCoverage",
                "-t",
                "5",
                "-sd",
                str(tmp_path / "savedir"),
                "-as",
                str(dlls),
                str(csv),
            ]
        )


def test_main_missing_assembly_infos_is_an_argparse_error(
    run_main, tmp_path: Path
) -> None:
    pysymgym = tmp_path / "pysymgym"

    with pytest.raises(SystemExit):
        run_main(
            [
                "-s",
                "ExecutionTreeContributedCoverage",
                "-t",
                "5",
                "-ps",
                str(pysymgym),
                "-sd",
                str(tmp_path / "savedir"),
            ]
        )
