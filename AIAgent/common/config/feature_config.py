from pathlib import Path

from paths import REPORT_PATH
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

_CONFIG = {"extra": "forbid", "populate_by_name": True}


@pydantic_dataclass(config=_CONFIG)
class SaveIfFailOrTimeout:
    """The save-on-failure/timeout feature's settings (data only).

    The actual model-saving behavior lives in ``ml.model_io.save_state_dict``
    (it needs torch, which the dependency-light ``common`` layer must not
    import). ``save_path`` defaults to the runtime artifact directory, keeping
    ``paths.REPORT_PATH`` the single source of truth for that location.
    """

    enabled: bool = Field(default=True)
    timeout_sec: int = Field(default=1800)
    save_path: Path = Field(default_factory=lambda: REPORT_PATH / "timeouted_agents")


@pydantic_dataclass(config=_CONFIG)
class OnGameServerRestartFeature:
    """The game-server restart feature's settings."""

    enabled: bool = Field(default=True)
    wait_for_reset_retries: int = Field(default=600)
    wait_for_reset_time: float = Field(default=0.1)


@pydantic_dataclass(config=_CONFIG)
class FeatureConfig:
    """Production-behavior feature flags and their settings.

    These were the mutable class-attribute singletons in the old
    ``AIAgent/config.py``; they are now YAML-configurable fields with the
    previous values as defaults.
    """

    verbose_tables: bool = Field(default=True)
    disable_message_checks: bool = Field(default=False)
    save_if_fail_or_timeout: SaveIfFailOrTimeout = Field(
        default_factory=SaveIfFailOrTimeout
    )
    on_game_server_restart: OnGameServerRestartFeature = Field(
        default_factory=OnGameServerRestartFeature
    )
    # Pin the game server's .NET runtime to the major.minor its
    # runtimeconfig.json declares (the dataset is generated with .NET 8):
    # an ambient DOTNET_ROLL_FORWARD must not roll the server forward to a
    # different major version, which changes the path-condition encoding
    # (#669). LatestPatch takes the highest installed patch of that
    # major.minor and disables minor/major roll-forward; Disable would bind
    # to the exact declared patch (8.0.0) and fail on any 8.0.x install.
    dotnet_roll_forward: str = Field(default="LatestPatch")
