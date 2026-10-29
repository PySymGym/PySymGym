"""Protocols shared across the agent packages.

Dependency-light on purpose: the shared layer (``common``) must stay
importable without the ML stack, so the interfaces it defines live here,
not in ``ml.protocols``.
"""

from abc import ABC, abstractmethod


class Named(ABC):
    """An object that reports its name (agents, predictors)."""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
