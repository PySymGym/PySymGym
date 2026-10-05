"""Unit tests for the message deserialization helpers in ``unsafe_json``."""

from dataclasses import dataclass

import pytest
from dataclasses_json import dataclass_json

from connection.game_server_conn.unsafe_json import asdict, obj_from_dict

pytestmark = pytest.mark.unit


@dataclass_json
@dataclass
class _Serializable:
    value: int


def test_obj_from_dict_builds_nested_attribute_objects() -> None:
    obj = obj_from_dict({"a": 1, "b": {"c": [1, {"d": 2}]}})
    assert obj.a == 1
    assert obj.b.c[0] == 1
    assert obj.b.c[1].d == 2


def test_asdict_inverts_obj_from_dict() -> None:
    data = {"a": 1, "b": {"c": [1, {"d": 2}]}}
    assert asdict(obj_from_dict(data)) == data


def test_asdict_delegates_to_to_json_for_serializable_dataclasses() -> None:
    assert asdict(_Serializable(value=3)) == _Serializable(value=3).to_json()


@pytest.mark.parametrize("value", [1, 1.5, "text"])
def test_asdict_passes_scalars_through(value: int | float | str) -> None:
    assert asdict(value) == value
