"""Unit tests for the runstrat test-runner invocation."""

import signal
import subprocess

import func_timeout
import pytest
from runstrat.psstrategy import AIStrategy, ExecutionTreeContributedCoverageStrategy
from runstrat.structs import LaunchInfo
from runstrat.subprocess_calls import call_test_runner

pytestmark = pytest.mark.unit


class FakeProc:
    """A stand-in for subprocess.Popen with a scripted communicate()."""

    pid = 4242

    def __init__(self, communicate_result=(b"runner output", None), returncode=0):
        self.communicate_result = communicate_result
        self.returncode = returncode
        self.waited = False

    def communicate(self, timeout=None):
        return self.communicate_result

    def wait(self):
        self.waited = True


class TimingOutProc(FakeProc):
    """A proc whose communicate() times out."""

    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="dotnet", timeout=timeout)


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch subprocess.Popen in the module; return the call/proc state.

    Register the proc(s) to hand out in ``state["procs"]`` (popped in
    order); each is tagged with the call and kwargs it was launched with.
    """
    state: dict = {"calls": [], "procs": []}

    def fake_popen(call, **kwargs):
        state["calls"].append((call, kwargs))
        proc = state["procs"].pop(0)
        proc.call = call
        proc.kwargs = kwargs
        return proc

    monkeypatch.setattr("runstrat.subprocess_calls.subprocess.Popen", fake_popen)
    return state


def test_call_test_runner_builds_execution_tree_argv(fake_popen: dict) -> None:
    fake_popen["procs"].append(FakeProc())
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")
    strategy = ExecutionTreeContributedCoverageStrategy(
        "ExecutionTreeContributedCoverage"
    )

    command, output = call_test_runner("/runner.dll", launch_info, strategy, "/work", 5)

    assert output == "runner output"
    call, kwargs = fake_popen["calls"][0]
    assert call == [
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
    # The child leads its own session so the group can be killed on timeout.
    assert kwargs == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "cwd": "/work",
        "start_new_session": True,
    }
    assert (
        command == "dotnet /runner.dll --method Cls.M /dlls/a.dll --timeout 5 "
        "--strat ExecutionTreeContributedCoverage --check-coverage"
    )


def test_call_test_runner_appends_model_for_ai_strategy(fake_popen: dict) -> None:
    fake_popen["procs"].append(FakeProc())
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")

    call_test_runner(
        "/runner.dll", launch_info, AIStrategy("AI", "model.onnx"), "/work", 5
    )

    call, kwargs = fake_popen["calls"][0]
    assert call[-2:] == ["--model", "model.onnx"]
    assert kwargs["cwd"] == "/work"


def test_call_test_runner_propagates_called_process_error(fake_popen: dict) -> None:
    fake_popen["procs"].append(
        FakeProc(communicate_result=(b"boom", None), returncode=1)
    )
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")
    strategy = ExecutionTreeContributedCoverageStrategy(
        "ExecutionTreeContributedCoverage"
    )

    # A nonzero exit surfaces as a CalledProcessError carrying the output
    # and the cmd, the fields the entrypoint logs.
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        call_test_runner("/runner.dll", launch_info, strategy, "/work", 5)

    assert exc_info.value.output == b"boom"
    assert exc_info.value.cmd == fake_popen["calls"][0][0]


def test_call_test_runner_kills_the_process_group_on_timeout(
    fake_popen: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    killpg_calls: list = []
    monkeypatch.setattr(
        "runstrat.subprocess_calls.os.killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    proc = TimingOutProc()
    fake_popen["procs"].append(proc)
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")
    strategy = ExecutionTreeContributedCoverageStrategy(
        "ExecutionTreeContributedCoverage"
    )

    with pytest.raises(func_timeout.FunctionTimedOut) as exc_info:
        call_test_runner("/runner.dll", launch_info, strategy, "/work", 5)

    # The whole group (dotnet may spawn children) is killed and reaped, and
    # the entrypoint's handler keeps getting the cmd via timedOutKwargs.
    assert killpg_calls == [(4242, signal.SIGKILL)]
    assert proc.waited is True
    assert exc_info.value.timedOutKwargs == {
        "call": fake_popen["calls"][0][0],
        "wdir": "/work",
    }


def test_call_test_runner_timeout_tolerates_an_already_dead_group(
    fake_popen: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dead_killpg(pgid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr("runstrat.subprocess_calls.os.killpg", dead_killpg)

    fake_popen["procs"].append(TimingOutProc())
    launch_info = LaunchInfo(dll="/dlls/a.dll", method="Cls.M")
    strategy = ExecutionTreeContributedCoverageStrategy(
        "ExecutionTreeContributedCoverage"
    )

    with pytest.raises(func_timeout.FunctionTimedOut):
        call_test_runner("/runner.dll", launch_info, strategy, "/work", 5)
