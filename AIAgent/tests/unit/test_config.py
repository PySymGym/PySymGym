"""Tests for the lazy device seam in ``config.get_device`` and the config
constant values.

The device must be resolved when it is used, not when ``config`` is imported,
so importing AIAgent code never probes CUDA and tests can force CPU. The
constant values are pinned as the drift net for #585 (merge the constant
classes into the pydantic YAML system), which carries the current values over
as the pydantic defaults.
"""

import re
from pathlib import Path

import pytest
import torch

from config import (
    BrokerConfig,
    FeatureConfig,
    GameServerConnectorConfig,
    GeneralConfig,
    OnGameServerRestartFeature,
    SaveIfFailOrTimeout,
    WebsocketSourceLinks,
    get_device,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, "cuda:0"), (False, "cpu")],
)
def test_get_device_selects_available_backend(
    cuda_available: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert get_device() == torch.device(expected)


def test_device_is_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cuda:0")


def test_no_import_time_device_binding() -> None:
    assert not hasattr(GeneralConfig, "DEVICE")


def test_save_model_writes_timestamped_state_dict(tmp_path: Path) -> None:
    cfg = SaveIfFailOrTimeout(
        enabled=True, timeout_sec=1800, save_path=tmp_path / "saved"
    )
    model = torch.nn.Linear(2, 2)

    cfg.save_model(model, with_name="agent")

    files = list((tmp_path / "saved").glob("agent_*.pt"))
    assert len(files) == 1
    # str(datetime) format: YYYY-MM-DD HH:MM:SS.ffffff (wall-clock, so the
    # pattern is asserted, not the exact timestamp).
    assert re.fullmatch(
        r"agent_\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\.pt", files[0].name
    )
    loaded = torch.load(files[0], weights_only=True)
    assert set(loaded) == set(model.state_dict())
    for key, tensor in model.state_dict().items():
        assert torch.equal(loaded[key], tensor)


def test_broker_config_and_urls() -> None:
    assert BrokerConfig.BROKER_PORT == 35000
    assert WebsocketSourceLinks.GET_WS == "http://0.0.0.0:35000/get_ws"
    assert WebsocketSourceLinks.POST_WS == "http://0.0.0.0:35000/post_ws"


def test_feature_config_defaults() -> None:
    # #585 carries these values over as the pydantic defaults; the pins are
    # the drift net. save_path is CWD-relative (the #584 target) — the value
    # is pinned, the CWD-dependence is not asserted as correct.
    assert FeatureConfig.VERBOSE_TABLES is True
    assert FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT == SaveIfFailOrTimeout(
        enabled=True, timeout_sec=1800, save_path=Path("./report/timeouted_agents/")
    )
    assert FeatureConfig.ON_GAME_SERVER_RESTART == OnGameServerRestartFeature(
        enabled=True, wait_for_reset_retries=600, wait_for_reset_time=0.1
    )
    # Pins the game server's .NET runtime to the major.minor its
    # runtimeconfig.json declares (#669).
    assert FeatureConfig.DOTNET_ROLL_FORWARD == "LatestPatch"


def test_game_server_connector_config_values() -> None:
    # RESPONCE_TIMEOUT_SEC is currently derived as
    # SAVE_IF_FAIL_OR_TIMEOUT.timeout_sec + 1 (the #585-targeted derivation);
    # the value 1801 is what #585 keeps as the default. The derivation
    # formula is not asserted.
    assert GameServerConnectorConfig.CREATE_CONNECTION_TIMEOUT_SEC == 1
    assert GameServerConnectorConfig.WAIT_FOR_SOCKET_RECONNECTION_MAX_RETRIES == 600
    assert GameServerConnectorConfig.RESPONCE_TIMEOUT_SEC == 1801
    assert GameServerConnectorConfig.SKIP_UTF_VALIDATION is True


def test_disable_message_checks_defaults_to_strict_validation() -> None:
    # Strict validation is the default (#579); the flag is kept for
    # compatibility and set explicitly by the tests that pin the unchecked
    # (BareObject) path.
    assert FeatureConfig.DISABLE_MESSAGE_CHECKS is False
