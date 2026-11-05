"""Unit tests for the broker connection data classes."""

from dataclasses import FrozenInstanceError

import pytest

from connection.broker_conn.classes import ServerInstanceInfo

pytestmark = pytest.mark.unit


def test_server_instance_info_is_frozen_and_hashable() -> None:
    info = ServerInstanceInfo(svm_name="svm", port=1, ws_url="ws://host", pid=2)
    assert hash(info)

    with pytest.raises(FrozenInstanceError):
        info.port = 3  # type: ignore[misc]
