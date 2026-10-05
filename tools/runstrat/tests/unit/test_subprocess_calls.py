"""Unit tests for the runstrat test-runner invocation."""

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
