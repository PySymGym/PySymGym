"""Dataset vocabulary shared across the agent packages.

Dependency-light on purpose: the shared layer (``common``) must stay
importable without the ML stack, so the dataset types it defines live
here, not in ``ml.dataset``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence, Tuple


class TrainingDatasetMode(Enum):
    """Which split of the training dataset is active.

    The member *values* are serialized into configs and MLflow artifacts,
    so they are pinned behaviour, not implementation detail.
    """

    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"


@dataclass(frozen=True)
class Result:
    """A map's validation result, ordered so that better results sort first.

    Better means: higher coverage, then fewer tests, then fewer steps, then
    fewer errors. The ordering lives solely in :attr:`sort_key` and
    ``__lt__``; construction sites pass the plain (non-negated) counts.

    The on-disk representation (:meth:`to_tuple`/:meth:`from_tuple`) keeps
    the historical sign-flipped tuple format
    ``(coverage, -tests, -steps, errors)`` so existing processed datasets
    remain readable.
    """

    coverage_percent: float
    tests_number: int
    steps_number: int
    errors_number: int

    @property
    def sort_key(self) -> Tuple[float, int, int, int]:
        """The lexicographic key implementing the ordering above (larger = better)."""
        return (
            self.coverage_percent,
            -self.tests_number,
            -self.steps_number,
            -self.errors_number,
        )

    def __lt__(self, other: "Result") -> bool:
        """Return whether this result sorts before ``other`` (i.e. ``other`` is better)."""
        if not isinstance(other, Result):
            return NotImplemented
        return self.sort_key < other.sort_key

    def __iter__(self) -> Iterator[float | int]:
        """Yield the four counts in field order."""
        yield from (
            self.coverage_percent,
            self.tests_number,
            self.steps_number,
            self.errors_number,
        )

    def to_tuple(self) -> Tuple[float, int, int, int]:
        """Return the on-disk representation: tests and steps negated."""
        return (
            self.coverage_percent,
            -self.tests_number,
            -self.steps_number,
            self.errors_number,
        )

    @classmethod
    def from_tuple(cls, values: Sequence[float | int]) -> "Result":
        """Build a :class:`Result` from its on-disk (sign-flipped) tuple."""
        coverage_percent, tests_number, steps_number, errors_number = values
        return cls(
            coverage_percent,
            -tests_number,
            -steps_number,
            errors_number,
        )
