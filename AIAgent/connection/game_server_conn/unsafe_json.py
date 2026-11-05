"""Tolerant decoding of the game-server wire protocol's JSON.

The unchecked decode path (``FeatureConfig.disable_message_checks``) turns a
parsed JSON document into plain objects whose fields are read as attributes
(``message.MessageType``). JSON objects become :class:`BareObject` instances
(their ``__dict__`` is the decoded mapping); JSON arrays and scalars pass
through unchanged, so a nested document keeps its shape.
"""

from typing import Any, overload


class BareObject:
    """An attribute bag for one decoded JSON object.

    ``obj_from_dict`` fills the instance's ``__dict__`` with the decoded
    mapping, so the object's fields are read as attributes. A missing field
    raises ``AttributeError`` — the default lookup outcome — via
    ``__getattr__``; declaring it lets the type checkers see that attribute
    access on a :class:`BareObject` is well-defined.
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


#: The shape of a decoded JSON document: objects become :class:`BareObject`,
#: arrays and scalars keep their Python shape.
type JsonValue = BareObject | list[JsonValue] | str | int | float | bool | None


@overload
def obj_from_dict(data: dict) -> BareObject: ...
@overload
def obj_from_dict(
    data: list | str | int | float | bool | None,
) -> list[JsonValue] | str | int | float | bool | None: ...


def obj_from_dict(
    data: dict | list | str | int | float | bool | None,
) -> JsonValue:
    if isinstance(data, list):
        return [obj_from_dict(item) for item in data]
    if isinstance(data, dict):
        inner_dict = {
            field_name: obj_from_dict(field_raw)
            for field_name, field_raw in data.items()
        }
        bare_obj = BareObject()
        bare_obj.__dict__ = inner_dict
        return bare_obj
    return data
