"""Unit tests for ``common.utils.inheritors``."""

import pytest

from common.utils import inheritors

pytestmark = pytest.mark.unit


class _Base:
    pass


class _Child(_Base):
    pass


class _GrandChild(_Child):
    pass


def test_inheritors_returns_all_transitive_subclasses() -> None:
    assert inheritors(_Base) == {_Child, _GrandChild}


def test_inheritors_of_leaf_class_is_empty() -> None:
    assert inheritors(_GrandChild) == set()
