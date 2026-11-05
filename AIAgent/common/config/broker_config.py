from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class BrokerConfig:
    """The websocket broker's listen port, the client-facing URLs, and the
    instance lease.

    The URLs are derived from ``port`` (the broker serves ``/get_ws`` and
    ``/post_ws`` on it), so a single field is the source of truth for both.

    ``lease_ttl_sec`` is how long an issued instance may be held before the
    broker reaps it: a hard-killed worker never calls ``/post_ws``, so without
    a lease its instance (and game-server process) would leak forever. The
    default (1 h) is larger than the per-map timeout (1800 s), so a
    legitimate game is never reaped, while a leaked instance is bounded to 1 h.
    """

    port: int = Field(default=35000)
    lease_ttl_sec: float = Field(default=3600)

    @property
    def get_ws_url(self) -> str:
        return f"http://0.0.0.0:{self.port}/get_ws"

    @property
    def post_ws_url(self) -> str:
        return f"http://0.0.0.0:{self.port}/post_ws"
