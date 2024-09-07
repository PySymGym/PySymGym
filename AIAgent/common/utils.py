from typing import TypeVar

T = TypeVar("T")


def inheritors(cls: T) -> set[T]:
    subclasses: set[T] = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses
