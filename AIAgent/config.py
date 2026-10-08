import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


class GeneralConfig:
    LOGGER_LEVEL = logging.INFO
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BrokerConfig:
    BROKER_PORT = 35000


class WebsocketSourceLinks:
    GET_WS = f"http://0.0.0.0:{BrokerConfig.BROKER_PORT}/get_ws"
    POST_WS = f"http://0.0.0.0:{BrokerConfig.BROKER_PORT}/post_ws"


@dataclass(slots=True, frozen=True)
class SaveIfFailOrTimeout:
    enabled: bool
    timeout_sec: int
    save_path: Path

    def save_model(self, model: torch.nn.Module, with_name: str):
        self.save_path.mkdir(exist_ok=True)
        timestamp = datetime.fromtimestamp(datetime.now().timestamp())

        torch.save(model.state_dict(), self.save_path / f"{with_name}_{timestamp}.pt")


@dataclass(slots=True, frozen=True)
class OnGameServerRestartFeature:
    enabled: bool
    wait_for_reset_retries: int
    wait_for_reset_time: float


class FeatureConfig:
    VERBOSE_TABLES = True
    DISABLE_MESSAGE_CHECKS = True
    SAVE_IF_FAIL_OR_TIMEOUT = SaveIfFailOrTimeout(
        enabled=True, timeout_sec=1800, save_path=Path("./report/timeouted_agents/")
    )
    ON_GAME_SERVER_RESTART = OnGameServerRestartFeature(
        enabled=True, wait_for_reset_retries=10 * 60, wait_for_reset_time=0.1
    )


class GameServerConnectorConfig:
    CREATE_CONNECTION_TIMEOUT_SEC = 1
    WAIT_FOR_SOCKET_RECONNECTION_MAX_RETRIES = 10 * 60
    RESPONCE_TIMEOUT_SEC = (
        FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.timeout_sec + 1
        if FeatureConfig.SAVE_IF_FAIL_OR_TIMEOUT.enabled
        else 1000
    )
    SKIP_UTF_VALIDATION = True
