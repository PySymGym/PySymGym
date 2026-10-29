from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class BrokerConfig:
    """The websocket broker's listen port and the client-facing URLs.

    The URLs are derived from ``port`` (the broker serves ``/get_ws`` and
    ``/post_ws`` on it), so a single field is the source of truth for both.
    """

    port: int = Field(default=35000)

    @property
    def get_ws_url(self) -> str:
        return f"http://0.0.0.0:{self.port}/get_ws"

    @property
    def post_ws_url(self) -> str:
        return f"http://0.0.0.0:{self.port}/post_ws"
