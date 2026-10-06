"""Shared fixtures for the AIAgent test suites.

These fixtures are the single place where AIAgent tests get a deterministic,
CPU-only, CWD-independent environment. They build real project objects through
the production code paths (``convert_input_to_tensor``, ``TrainingDataset``)
instead of hand-rolled copies, so they cannot drift from the real schema.
"""

import random
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from common.game import (
    GameEdgeLabel,
    GameMap,
    GameMap2SVM,
    GameMapEdge,
    GameMapVertex,
    GameState,
    PathConditionVertex,
    State,
    StateHistoryElem,
)
from common.validation_coverage.svm_info import SVMInfo
from ml.dataset import NUM_STATE_FEATURES, TrainingDataset, convert_input_to_tensor
from ml.inference import TORCH

DEFAULT_GAME_FEATURES = 7
RANDOM_SEED = 0


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``config.get_device()`` resolve to CPU so tests never use a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def seeded_rng() -> None:
    """Seed every RNG the suite uses so fixtures and models are deterministic."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


@pytest.fixture
def gamestate_factory():
    """Return a factory building a small, fully-connected synthetic gamestate.

    The graph has 3 system vertices, 2 states and 2 path-condition vertices,
    wired so that every node and edge type produced by
    :func:`ml.dataset.convert_input_to_tensor` is non-empty.
    """

    def make() -> GameState:
        vertices = [
            GameMapVertex(
                Id=0,
                InCoverageZone=True,
                BasicBlockSize=4,
                CoveredByTest=True,
                VisitedByState=True,
                TouchedByState=True,
                ContainsCall=False,
                ContainsThrow=False,
                States=[0],
            ),
            GameMapVertex(
                Id=1,
                InCoverageZone=True,
                BasicBlockSize=5,
                CoveredByTest=True,
                VisitedByState=True,
                TouchedByState=True,
                ContainsCall=False,
                ContainsThrow=False,
                States=[0, 1],
            ),
            GameMapVertex(
                Id=2,
                InCoverageZone=False,
                BasicBlockSize=6,
                CoveredByTest=False,
                VisitedByState=True,
                TouchedByState=False,
                ContainsCall=True,
                ContainsThrow=False,
                States=[1],
            ),
        ]
        states = [
            State(
                Id=0,
                Position=0,
                PathCondition=[0],
                VisitedAgainVertices=0,
                VisitedNotCoveredVerticesInZone=0,
                VisitedNotCoveredVerticesOutOfZone=0,
                History=[
                    StateHistoryElem(
                        GraphVertexId=0, NumOfVisits=1, StepWhenVisitedLastTime=0
                    )
                ],
                Children=[1],
                StepWhenMovedLastTime=0,
                InstructionsVisitedInCurrentBlock=1,
            ),
            State(
                Id=1,
                Position=1,
                PathCondition=[1],
                VisitedAgainVertices=1,
                VisitedNotCoveredVerticesInZone=0,
                VisitedNotCoveredVerticesOutOfZone=1,
                History=[
                    StateHistoryElem(
                        GraphVertexId=1, NumOfVisits=1, StepWhenVisitedLastTime=1
                    )
                ],
                Children=[],
                StepWhenMovedLastTime=1,
                InstructionsVisitedInCurrentBlock=2,
            ),
        ]
        path_condition_vertices = [
            PathConditionVertex(Id=0, Type=0, Children=[1]),
            PathConditionVertex(Id=1, Type=1, Children=[]),
        ]
        edges = [
            GameMapEdge(VertexFrom=0, VertexTo=1, Label=GameEdgeLabel(Token=0)),
            GameMapEdge(VertexFrom=1, VertexTo=2, Label=GameEdgeLabel(Token=1)),
        ]
        return GameState(
            GraphVertices=vertices,
            States=states,
            PathConditionVertices=path_condition_vertices,
            Map=edges,
        )

    return make


@pytest.fixture
def hetero_factory(gamestate_factory):
    """Return a factory building the graph input a model forward expects.

    Parameters
    ----------
    game_features : int, optional
        Width of the game-vertex feature matrix (7 for current models, 5 for
        the legacy ones).
    state_features : int, optional
        Width of the state-vertex feature matrix.
    """

    def make(
        game_features: int = DEFAULT_GAME_FEATURES,
        state_features: int = NUM_STATE_FEATURES,
    ) -> HeteroData:
        data, _ = convert_input_to_tensor(gamestate_factory())
        data[TORCH.game_vertex].x = data[TORCH.game_vertex].x[:, :game_features]
        data[TORCH.state_vertex].x = data[TORCH.state_vertex].x[:, :state_features]
        return data

    return make


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> TrainingDataset:
    """An empty :class:`TrainingDataset` rooted in a temporary directory."""
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    processed_dir.mkdir()
    raw_dir.mkdir()
    return TrainingDataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_percentage=1.0,
        n_jobs=1,
    )


class FakeWebSocket:
    """A recording websocket stand-in that never touches a real socket.

    Outgoing frames are appended to :attr:`sent`; :meth:`recv` pops the next
    frame from :attr:`incoming` (raising it instead when it is an exception).
    Set :attr:`will_connect` to ``False`` to simulate a connection attempt that
    never completes, and :attr:`connect_error` to raise on ``connect``.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.incoming: list[object] = []
        self.connected = False
        self.will_connect = True
        self.connect_error: BaseException | None = None
        self.timeout: float | None = None
        self.closed = False
        self.url: str | None = None

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def connect(self, url: str, skip_utf8_validation: bool = True) -> None:
        self.url = url
        if self.connect_error is not None:
            raise self.connect_error
        if self.will_connect:
            self.connected = True

    def send(self, message: str) -> None:
        self.sent.append(message)

    def recv(self) -> str:
        if not self.incoming:
            raise AssertionError("no incoming websocket frame was queued")
        frame = self.incoming.pop(0)
        if isinstance(frame, BaseException):
            raise frame
        return frame  # type: ignore[return-value]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_websocket() -> FakeWebSocket:
    """Return a fresh :class:`FakeWebSocket` recording frames, no real socket."""
    return FakeWebSocket()


class FakeNamespace:
    """The multiprocessing namespace attributes the game managers read."""

    def __init__(self) -> None:
        self.shared_lock = threading.Lock()
        self.is_prepared = SimpleNamespace(value=False)


class FakeProcess:
    """A ``subprocess.Popen`` stand-in recording kill/communicate calls.

    ``poll_result`` is what :meth:`poll` returns; ``None`` means the process is
    still running. Replaces a real game-server process in unit tests.
    """

    def __init__(
        self,
        poll_result: int | None = None,
        output: tuple[str, str] = ("stdout", "stderr"),
        pid: int = 4242,
    ) -> None:
        self.pid = pid
        self._poll_result = poll_result
        self._output = output
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self._poll_result

    def wait(self) -> int:
        self.waited = True
        return 0

    def communicate(self) -> tuple[str, str]:
        return self._output

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def fake_namespace() -> FakeNamespace:
    """Return a fresh :class:`FakeNamespace` for the game-manager tests."""
    return FakeNamespace()


@pytest.fixture
def fake_proc():
    """Return a factory for :class:`FakeProcess` instances."""

    def make(**kwargs) -> FakeProcess:
        return FakeProcess(**kwargs)

    return make


@pytest.fixture
def game_map_factory():
    """Return a factory building a :class:`GameMap` with sensible defaults."""

    def make(
        map_name: str = "Method_0",
        steps_to_play: int = 10,
        steps_to_start: int = 0,
    ) -> GameMap:
        return GameMap(
            StepsToPlay=steps_to_play,
            StepsToStart=steps_to_start,
            AssemblyFullName="assembly",
            NameOfObjectToCover="Method",
            DefaultSearcher="BFS",
            MapName=map_name,
        )

    return make


@pytest.fixture
def game_map2svm_factory(game_map_factory):
    """Return a factory pairing a :class:`GameMap` with a synthetic ``SVMInfo``."""

    def make(
        map_name: str = "Method_0",
        steps_to_play: int = 10,
        steps_to_start: int = 0,
    ) -> GameMap2SVM:
        return GameMap2SVM(
            GameMap=game_map_factory(map_name, steps_to_play, steps_to_start),
            SVMInfo=SVMInfo(
                name="svm",
                launch_command="run",
                server_working_dir="/tmp",
                min_port=1,
                max_port=2,
            ),
        )

    return make
