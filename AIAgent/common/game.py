from dataclasses import dataclass

from common.validation_coverage.svm_info import SVMInfo
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(slots=True)
class StateHistoryElem:
    GraphVertexId: int
    NumOfVisits: int
    StepWhenVisitedLastTime: int


@dataclass_json
@dataclass(slots=True)
class State:
    Id: int
    Position: int
    PathConditionSize: int
    VisitedAgainVertices: int
    VisitedNotCoveredVerticesInZone: int
    VisitedNotCoveredVerticesOutOfZone: int
    History: list[StateHistoryElem]
    Children: list[int]
    StepWhenMovedLastTime: int
    InstructionsVisitedInCurrentBlock: int

    def __hash__(self) -> int:
        return self.Id.__hash__()


@dataclass_json
@dataclass(slots=True)
class GameMapVertex:
    Id: int
    InCoverageZone: bool
    BasicBlockSize: int
    CoveredByTest: bool
    VisitedByState: bool
    TouchedByState: bool
    ContainsCall: bool
    ContainsThrow: bool
    States: list[int]


@dataclass_json
@dataclass(slots=True)
class GameEdgeLabel:
    Token: int


@dataclass_json
@dataclass(slots=True)
class GameMapEdge:
    VertexFrom: int
    VertexTo: int
    Label: GameEdgeLabel


@dataclass_json
@dataclass(slots=True)
class GameState:
    GraphVertices: list[GameMapVertex]
    States: list[State]
    Map: list[GameMapEdge]


@dataclass_json
@dataclass(slots=True)
class GameMap:
    StepsToPlay: int
    StepsToStart: int
    AssemblyFullName: str
    NameOfObjectToCover: str
    DefaultSearcher: str
    MapName: str


@dataclass(slots=True)
class GameMap2SVM:
    GameMap: GameMap
    SVMInfo: SVMInfo


@dataclass_json
@dataclass(slots=True)
class MoveReward:
    ForCoverage: int
    ForVisitedInstructions: int

    def printable(self, verbose=False) -> str:
        if verbose:
            return f"ForVisitedInstructions: {self.ForVisitedInstructions}"
        return f"#vi={self.ForVisitedInstructions}"


@dataclass_json
@dataclass(slots=True)
class Reward:
    ForMove: MoveReward
    MaxPossibleReward: int
