"""Unit tests for the broker connection data classes."""

from dataclasses import FrozenInstanceError

import pytest

from config import FeatureConfig
from connection.broker_conn.classes import (
    ServerInstanceInfo,
    custom_encoder_if_disable_message_checks,
)
from connection.game_server_conn.unsafe_json import asdict

pytestmark = pytest.mark.unit


def test_server_instance_info_is_frozen_and_hashable() -> None:
    info = ServerInstanceInfo(svm_name="svm", port=1, ws_url="ws://host", pid=2)
    assert hash(info)

    with pytest.raises(FrozenInstanceError):
        info.port = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("disabled", "expected"),
    [(True, asdict), (False, None)],
)
def test_custom_encoder_follows_feature_flag(
    disabled: bool, expected, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FeatureConfig, "DISABLE_MESSAGE_CHECKS", disabled)
    assert custom_encoder_if_disable_message_checks() is expected
