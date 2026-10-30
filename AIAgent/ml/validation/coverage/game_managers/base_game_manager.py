import logging
from abc import ABC, abstractmethod
from multiprocessing.managers import Namespace
from typing import ClassVar, Optional

from common.classes import GameFailed, Map2Result
from common.config.feature_config import FeatureConfig
from common.game import GameMap, GameMap2SVM
from torch_geometric.data.hetero_data import HeteroData


class BaseGamePreparator(ABC):
    """Abstract class for preparing the game environment before starting the game.

    This class provides a basic structure for performing preparatory actions
    required before starting the game process. The preparation is executed
    in a thread-safe manner, ensuring that the process is performed only once.
    """

    def __init__(self, namespace: Namespace):
        """
        Initializes the base game preparator.

        Attributes:
            _is_prepared: Flag indicating whether the preparation is complete.
            _shared_lock: Thread lock to ensure thread-safe preparation.
        """
        self._shared_lock = namespace.shared_lock
        self._is_prepared = namespace.is_prepared

    def prepare(self):
        """
        Ensures that the preparation process is executed only once in a thread-safe manner.

        This method allows only one thread to perform the preparation and blocks
        all attempts after the first thread completes it.
        """
        with self._shared_lock:
            if self._is_prepared.value:
                return
            self._prepare()
            self._is_prepared.value = True
            return

    @abstractmethod
    def _prepare(self):
        """
        Prepares the game environment before starting the game.
        """
        ...


class BaseGameManager(ABC):
    """Base class for the coverage-validation game managers.

    Owns the prepare-once lifecycle and the failure/timeout handling shared by
    every manager: building a ``GameFailed`` whose reason names the exception
    (not the class, #673) and the structured per-map timeout message (#616).
    Subclasses implement the mode-specific play loop and artifact handling.
    """

    #: The validation mode name, used in structured failure/timeout logs (#616).
    mode: ClassVar[str]

    def __init__(self, namespace: Namespace, *, feature_config: FeatureConfig):
        self._namespace = namespace
        # The sub-config is pickled to the spawn-based mp.Pool workers together
        # with the manager, so it arrives as an instance attribute, never a
        # process global.
        self._feature_config = feature_config
        self._preparator = self._create_preparator()

    def play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        """Executes GameMap on symbolic engine"""
        self._preparator.prepare()
        return self._play_game_map(game_map2svm)

    # -- shared failure/timeout handling ---------------------------------

    @staticmethod
    def _failure_reason(error: BaseException) -> str:
        """The ``GameFailed`` reason for an unexpected error (#673).

        Carries the exception type and a human-readable message so consumers
        can distinguish failure causes; the old code stored the exception class
        object itself (e.g. ``<class 'TypeError'>``), which carried no message.
        ``GameInterruptedError`` subclasses expose their message via ``desc``
        (they are constructed without one), so prefer that over ``str``.
        """
        message = getattr(error, "desc", None) or str(error)
        return f"{type(error).__name__}: {message}"

    def _timeout_message(
        self,
        *,
        map_name: str,
        steps_budget: int,
        elapsed: float,
        last_step: Optional[int] = None,
    ) -> str:
        """A structured per-map timeout message (#616): no raw args reprs.

        Names the map, the mode, the step budget, the elapsed time, and the
        last observed step (when the mode observes steps) so a timed-out map is
        diagnosable from the log alone.
        """
        last = f" at step {last_step}/{steps_budget}" if last_step is not None else ""
        return (
            f"[{self.mode}] map {map_name} timed out after {elapsed:.1f}s "
            f"(step budget {steps_budget}){last}"
        )

    def _timeout_failure(
        self,
        *,
        map_name: str,
        steps_budget: int,
        elapsed: float,
        last_step: Optional[int] = None,
    ) -> GameFailed:
        """Log the structured timeout message at ERROR and build the failure."""
        message = self._timeout_message(
            map_name=map_name,
            steps_budget=steps_budget,
            elapsed=elapsed,
            last_step=last_step,
        )
        logging.error(message)
        return GameFailed(reason=message)

    @abstractmethod
    def _create_preparator(self) -> BaseGamePreparator: ...

    @abstractmethod
    def _play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        """Executes GameMap on symbolic engine.

        This method is called by 'play_game_map' after the assurance that preparation step is not skipped.
        """
        ...

    @abstractmethod
    def get_game_steps(self, game_map: GameMap) -> Optional[list[HeteroData]]:
        """Returns list of HeteroData for each step. Returns None if steps of game_map can't be received."""
        ...

    @abstractmethod
    def delete_game_artifacts(self, game_map: GameMap):
        """Deletes game artifacts of game_map"""
        ...

    @abstractmethod
    def notify_steps_requirement(self, game_map: GameMap, required: bool):
        """Notify the symbolic execution environment whether to preserve execution steps.

        Allows the client to optimize resource usage by skipping unnecessary step data
        transmission when metrics are not improved.

        Args:
            game_map: game map
            required: True if steps should be preserved and transmitted
        """
        ...
