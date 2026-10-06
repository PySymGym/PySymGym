"""Small helpers shared across the agent packages."""

from typing import TypeVar

T = TypeVar("T")


def inheritors(cls: T) -> set[T]:
    """Return every transitive subclass of ``cls``.

    Parameters
    ----------
    cls : T
        The base class to collect subclasses of.

    Returns
    -------
    set[T]
        All direct and indirect subclasses of ``cls``.
    """
    subclasses: set[T] = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses
