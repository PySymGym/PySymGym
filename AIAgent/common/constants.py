from pathlib import Path

from config import BrokerConfig
import os


def _build_bar_format() -> str:
    custom_left = "{desc}: {n_fmt}/{total_fmt}"
    custom_bar = "{percentage:3.0f}% [{bar}]"
    custom_info = "{elapsed}<{remaining}, {rate_fmt}{postfix}"

    return f"{custom_left} {custom_bar} - {custom_info}"


BASE_REPORT_DIR = Path("./report")
TABLES_LOG_FILE = BASE_REPORT_DIR / "tables.log"
LEADERS_TABLES_LOG_FILE = BASE_REPORT_DIR / "leaders.log"
EPOCH_BEST_DIR = BASE_REPORT_DIR / "epochs_best"
APP_LOG_FILE = Path("app.log")

TQDM_FORMAT_DICT = {
    "unit": "game",
    "bar_format": _build_bar_format(),
    "dynamic_ncols": True,
}


class WebsocketSourceLinks:
    GET_WS = f"http://0.0.0.0:{BrokerConfig.BROKER_PORT}/get_ws"
    POST_WS = f"http://0.0.0.0:{BrokerConfig.BROKER_PORT}/post_ws"
