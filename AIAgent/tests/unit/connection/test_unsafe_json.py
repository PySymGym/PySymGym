"""Unit tests for the message deserialization helpers in ``unsafe_json``."""

import pytest

from connection.game_server_conn.unsafe_json import obj_from_dict

pytestmark = pytest.mark.unit


def test_obj_from_dict_builds_nested_attribute_objects() -> None:
    obj = obj_from_dict({"a": 1, "b": {"c": [1, {"d": 2}]}})
    assert obj.a == 1
    assert obj.b.c[0] == 1
    assert obj.b.c[1].d == 2
