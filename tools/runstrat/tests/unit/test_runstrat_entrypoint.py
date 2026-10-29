"""Seam tests for the runstrat ``entrypoint()`` (happy path + result writing).

``entrypoint()`` is driven with a method-keyed fake for
``runstrat.cli.call_test_runner`` (the subprocess boundary) and a recording
fake for ``logging.basicConfig`` (no global logging side effect).
``load_config`` and ``parse_runner_output`` run for real; ``tqdm`` is
replaced by a passthrough so the loop is deterministic.
"""

import logging
import subprocess
from pathlib import Path

import func_timeout
import pytest
import runstrat.cli
from runstrat import Args, PATH_TO_VSHARP, RunMode
from runstrat.psstrategy import ExecutionTreeContributedCoverageStrategy

pytestmark = pytest.mark.unit

# Parses to (total_time_sec, tests, errors, coverage) = (62, 5, 2, 88.5).
OUTPUT_M1 = (
    "Total time: 00:01:02.123\n"
    "Tests generated: 5\n"
    "Errors generated: 2\n"
    "Precise coverage: 88.5\n"
)
# Parses to (3600, 10, 0, 100.0).
OUTPUT_M2 = (
    "Total time: 01:00:00.000\n"
    "Tests generated: 10\n"
    "Errors generated: 0\n"
    "Precise coverage: 100.0\n"
)
# Missing the "Total time" line -> parse_runner_output raises AttributeError.
OUTPUT_NO_TIME = "Tests generated: 5\nErrors generated: 2\nPrecise coverage: 88.5\n"
# Trailing text after the coverage -> float() raises ValueError (not caught by
# entrypoint, which only catches AttributeError; #590-tracked).
OUTPUT_BAD_COVERAGE = (
    "Total time: 00:01:02.123\n"
    "Tests generated: 5\n"
    "Errors generated: 2\n"
    "Precise coverage: 88.5 extra\n"
)


def make_args(tmp_path: Path, methods: list[str], run_mode=RunMode.BENCHMARKS) -> Args:
    """Build an Args whose maps CSV yields one launch info per method."""
    csv_path = tmp_path / "maps.csv"
    csv_path.write_text("dll,method\n" + "".join(f"A.dll,{m}\n" for m in methods))
    dll_dir = tmp_path / "dlls"
    dll_dir.mkdir(exist_ok=True)
    savedir = tmp_path / "savedir"
    savedir.mkdir(exist_ok=True)
    # entrypoint() validates the runner up front; give it a file to find.
    runner = tmp_path / "pysymgym" / PATH_TO_VSHARP
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.touch()
    return Args(
        strategy=ExecutionTreeContributedCoverageStrategy(
            "ExecutionTreeContributedCoverage"
        ),
        timeout=5,
        pysymgym_path=tmp_path / "pysymgym",
        savedir=savedir,
        assembly_infos=[(dll_dir, csv_path)],
        run_mode=run_mode,
    )


@pytest.fixture
def entrypoint_env(monkeypatch: pytest.MonkeyPatch):
    """Install the entrypoint() fakes; return a dict with the script hook."""
    env: dict = {"script": {}, "basicConfig_calls": []}

    def fake_call_test_runner(path_to_runner, launch_info, strategy, wdir, timeout):
        entry = env["script"][launch_info.method]
        if isinstance(entry, BaseException):
            raise entry
        return f"dotnet {path_to_runner} --method {launch_info.method}", entry

    def fake_basicConfig(**kwargs):
        env["basicConfig_calls"].append(kwargs)

    def fake_tqdm(iterable, *args, **kwargs):
        return iterable

    monkeypatch.setattr(runstrat.cli, "call_test_runner", fake_call_test_runner)
    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)
    monkeypatch.setattr(runstrat.cli.tqdm, "tqdm", fake_tqdm)
    return env


def test_setup_logging_configures_a_file_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))

    runstrat.cli.setup_logging(tmp_path / "logs", "MyStrat")

    assert calls[0]["filename"] == str(tmp_path / "logs" / "MyStrat.log")
    assert calls[0]["level"] == logging.INFO


def test_entrypoint_missing_runner_raises_file_not_found(
    entrypoint_env, tmp_path: Path
) -> None:
    args = make_args(tmp_path, ["M1"])
    args.pysymgym_path = tmp_path / "missing"
    entrypoint_env["script"] = {}

    with pytest.raises(FileNotFoundError, match="V# runner not found at"):
        runstrat.cli.entrypoint(args)


def test_entrypoint_writes_one_row_per_method(entrypoint_env, tmp_path: Path) -> None:
    args = make_args(tmp_path, ["M1", "M2"])
    entrypoint_env["script"] = {"M1": OUTPUT_M1, "M2": OUTPUT_M2}

    df = runstrat.cli.entrypoint(args)

    assert list(df.columns) == [
        "method",
        "tests",
        "errors",
        "coverage",
        "total_time_sec",
    ]
    assert df["method"].tolist() == ["M1", "M2"]
    assert df["tests"].tolist() == [5, 10]
    assert df["errors"].tolist() == [2, 0]
    assert df["coverage"].tolist() == [88.5, 100.0]
    assert df["total_time_sec"].tolist() == [62, 3600]

    csv_path = tmp_path / "savedir" / "ExecutionTreeContributedCoverage.csv"
    assert csv_path.read_text() == (
        "method,tests,errors,coverage,total_time_sec\n"
        "M1,5,2,88.5,62\n"
        "M2,10,0,100.0,3600\n"
    )
    # setup_logging pointed the file handler at {strategy.name}.log.
    assert entrypoint_env["basicConfig_calls"][0]["filename"] == str(
        tmp_path / "savedir" / "ExecutionTreeContributedCoverage.log"
    )


def test_entrypoint_returns_empty_dataframe_when_no_methods(
    entrypoint_env, tmp_path: Path
) -> None:
    args = make_args(tmp_path, [])
    entrypoint_env["script"] = {}

    df = runstrat.cli.entrypoint(args)

    assert len(df) == 0
    assert list(df.columns) == []
    csv_path = tmp_path / "savedir" / "ExecutionTreeContributedCoverage.csv"
    assert csv_path.read_text() == "\n"


def _timed_out(method: str) -> func_timeout.FunctionTimedOut:
    # The entrypoint handler reads timedOutKwargs["call"] to log the cmd.
    return func_timeout.FunctionTimedOut(
        timedOutKwargs={"call": ["dotnet", "runner.dll", "--method", method]}
    )


def test_entrypoint_benchmarks_skips_failed_methods(
    entrypoint_env, tmp_path: Path
) -> None:
    args = make_args(tmp_path, ["CPE", "FTO", "AE", "OK"])
    entrypoint_env["script"] = {
        "CPE": subprocess.CalledProcessError(1, ["dotnet", "runner.dll"]),
        "FTO": _timed_out("FTO"),
        "AE": OUTPUT_NO_TIME,
        "OK": OUTPUT_M1,
    }

    df = runstrat.cli.entrypoint(args)

    # BENCHMARKS mode logs and skips each failure; only the successful method
    # is recorded.
    assert df["method"].tolist() == ["OK"]
    csv_path = tmp_path / "savedir" / "ExecutionTreeContributedCoverage.csv"
    assert csv_path.read_text() == (
        "method,tests,errors,coverage,total_time_sec\nOK,5,2,88.5,62\n"
    )


def test_entrypoint_debug_reraises_called_process_error(
    entrypoint_env, tmp_path: Path
) -> None:
    args = make_args(tmp_path, ["CPE"], run_mode=RunMode.DEBUG)
    entrypoint_env["script"] = {
        "CPE": subprocess.CalledProcessError(1, ["dotnet", "runner.dll"])
    }

    with pytest.raises(subprocess.CalledProcessError):
        runstrat.cli.entrypoint(args)


def test_entrypoint_debug_reraises_function_timed_out(
    entrypoint_env, tmp_path: Path
) -> None:
    args = make_args(tmp_path, ["FTO"], run_mode=RunMode.DEBUG)
    entrypoint_env["script"] = {"FTO": _timed_out("FTO")}

    with pytest.raises(func_timeout.FunctionTimedOut):
        runstrat.cli.entrypoint(args)


def test_entrypoint_debug_reraises_attribute_error(
    entrypoint_env, tmp_path: Path
) -> None:
    args = make_args(tmp_path, ["AE"], run_mode=RunMode.DEBUG)
    entrypoint_env["script"] = {"AE": OUTPUT_NO_TIME}

    with pytest.raises(AttributeError):
        runstrat.cli.entrypoint(args)


@pytest.mark.xfail(
    strict=True,
    reason="#590: a ValueError from float(total_coverage) is not caught by "
    "entrypoint (only AttributeError is), so one malformed line kills the run "
    "and discards all results; per-line parse errors should be skipped",
)
def test_entrypoint_benchmarks_skips_a_value_error_method(
    entrypoint_env, tmp_path: Path
) -> None:
    args = make_args(tmp_path, ["BAD", "OK"])
    entrypoint_env["script"] = {"BAD": OUTPUT_BAD_COVERAGE, "OK": OUTPUT_M1}

    df = runstrat.cli.entrypoint(args)

    assert df["method"].tolist() == ["OK"]
