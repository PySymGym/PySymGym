"""Shared fixtures for the AIAgent test suites.

These fixtures are the single place where AIAgent tests get a deterministic,
CPU-only, CWD-independent environment. They build real project objects through
the production code paths (``convert_input_to_tensor``, ``TrainingDataset``)
instead of hand-rolled copies, so they cannot drift from the real schema.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

import config
from common.game import (
    GameEdgeLabel,
    GameMapEdge,
    GameMapVertex,
    GameState,
    PathConditionVertex,
    State,
    StateHistoryElem,
)
from ml.dataset import NUM_STATE_FEATURES, TrainingDataset, convert_input_to_tensor
from ml.inference import TORCH

DEFAULT_GAME_FEATURES = 7
RANDOM_SEED = 0


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the project device to CPU so tests never allocate on a GPU host."""
    monkeypatch.setattr(config.GeneralConfig, "DEVICE", torch.device("cpu"))


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
