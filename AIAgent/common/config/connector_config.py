from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class GameServerConnectorConfig:
    """Timeouts and limits for the client's connection to a game server.

    Two independent budgets, both explicitly configurable:

    - The **connect-retry budget** is
      ``wait_for_socket_reconnection_max_retries`` ×
      (``create_connection_timeout_sec`` connect + the same sleep between
      attempts) — 60 × 2 s = 2 min by default. A dead server must not hold a
      worker for the old ~20 min.
    - The **recv timeout** (``response_timeout_sec``) bounds a single
      response from the game server. It is deliberately *smaller* than the
      per-map timeout (``SaveIfFailOrTimeout.timeout_sec``), so a dead
      connection is detected by the recv (60 s by default) instead of stalling
      a worker for ~30 min, and a hung recv self-terminates within that bound
      (which also bounds the lifetime of a timed-out game's worker thread).
      It is independent of the model-save feature: toggling that feature
      cannot silently change the socket recv timeout.
    """

    create_connection_timeout_sec: int = Field(default=1)
    wait_for_socket_reconnection_max_retries: int = Field(default=60)
    response_timeout_sec: int = Field(default=60)
    skip_utf_validation: bool = Field(default=True)
