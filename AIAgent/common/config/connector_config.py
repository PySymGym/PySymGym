from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class GameServerConnectorConfig:
    """Timeouts and limits for the client's connection to a game server.

    ``response_timeout_sec`` is an explicit, independently configurable field:
    it is no longer derived from the model-save feature's timeout, so toggling
    that feature cannot silently change the socket recv timeout.
    """

    create_connection_timeout_sec: int = Field(default=1)
    wait_for_socket_reconnection_max_retries: int = Field(default=600)
    response_timeout_sec: int = Field(default=1801)
    skip_utf_validation: bool = Field(default=True)
