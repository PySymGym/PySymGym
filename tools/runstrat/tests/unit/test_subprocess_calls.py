"""Unit tests for the runstrat test-runner invocation."""

import subprocess

import func_timeout
import pytest
from src.psstrategy import AIStrategy, ExecutionTreeContributedCoverageStrategy
from src.structs import LaunchInfo
from src.subprocess_calls import call_test_runner

pytestmark = pytest.mark.unit


@pytest.fixture
def captured_call(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def fake_check_output(call, **kwargs):
        calls.append((call, kwargs))
        return b"runner output"

    monkeypatch.setattr(
        "src.subprocess_calls.subprocess.check_output", fake_check_output
    )
    return calls


def test_call_test_runner_builds_execution_tree_argv(captured_call: list) -> None:
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")
    strategy = ExecutionTreeContributedCoverageStrategy(
        "ExecutionTreeContributedCoverage"
    )

    command, output = call_test_runner("/runner.dll", launch_info, strategy, "/work", 5)

    assert output == "runner output"
    assert captured_call[0][0] == [
        "dotnet",
        "/runner.dll",
        "--method",
        "Cls.M",
        "/dlls/a.dll",
        "--timeout",
        "5",
        "--strat",
        "ExecutionTreeContributedCoverage",
        "--check-coverage",
    ]
    assert (
        command == "dotnet /runner.dll --method Cls.M /dlls/a.dll --timeout 5 "
        "--strat ExecutionTreeContributedCoverage --check-coverage"
    )


def test_call_test_runner_appends_model_for_ai_strategy(captured_call: list) -> None:
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")

    call_test_runner(
        "/runner.dll", launch_info, AIStrategy("AI", "model.onnx"), "/work", 5
    )

    assert captured_call[0][0][-2:] == ["--model", "model.onnx"]
    assert captured_call[0][1]["cwd"] == "/work"


def test_call_test_runner_propagates_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_check_output(call, **kwargs):
        raise subprocess.CalledProcessError(1, call)

    monkeypatch.setattr(
        "src.subprocess_calls.subprocess.check_output", fake_check_output
    )
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")
    strategy = ExecutionTreeContributedCoverageStrategy(
        "ExecutionTreeContributedCoverage"
    )

    # The real func_timeout lets non-timeout exceptions through, so the
    # CalledProcessError raised by the (faked) check_output propagates out of
    # call_test_runner unchanged.
    with pytest.raises(subprocess.CalledProcessError):
        call_test_runner("/runner.dll", launch_info, strategy, "/work", 5)


def test_call_test_runner_propagates_function_timed_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_func_timeout(timeout, func, kwargs):
        raise func_timeout.FunctionTimedOut(
            timedOutKwargs={"call": ["dotnet", "/runner.dll"], "wdir": "/work"}
        )

    monkeypatch.setattr(
        "src.subprocess_calls.func_timeout.func_timeout", fake_func_timeout
    )
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")
    strategy = ExecutionTreeContributedCoverageStrategy(
        "ExecutionTreeContributedCoverage"
    )

    # The timeout is faked (no real sleeping/threading); call_test_runner
    # re-raises the FunctionTimedOut unchanged. The child-cleanup on timeout is
    # a #590-tracked defect and is not asserted here.
    with pytest.raises(func_timeout.FunctionTimedOut):
        call_test_runner("/runner.dll", launch_info, strategy, "/work", 5)
