import json
import logging
from urllib.parse import urlencode

import httplib2
from common.config.broker_config import BrokerConfig
from common.validation_coverage.svm_info import SVMInfo
from connection.broker_conn.classes import ServerInstanceInfo


def acquire_instance(
    svm_info: SVMInfo, broker_config: BrokerConfig
) -> ServerInstanceInfo:
    response, content = httplib2.Http().request(
        broker_config.get_ws_url + "?" + urlencode(SVMInfo.to_dict(svm_info))
    )
    if response.status != 200:
        logging.error(
            f"{response.status} with {content=} on acquire_instance call for {svm_info}"
        )
        raise RuntimeError(f"Not ok response: {response}, {content}")
    acquired_instance = ServerInstanceInfo.from_dict(
        json.loads(content.decode("utf-8"))
    )
    logging.info(f"acquired ws: {acquired_instance}")
    return acquired_instance


def return_instance(instance: ServerInstanceInfo, broker_config: BrokerConfig):
    logging.info(f"returning: {instance}")

    response, _ = httplib2.Http().request(
        broker_config.post_ws_url,
        method="POST",
        body=instance.to_json(),
    )

    if response.status == 200:
        logging.info(f"{instance} is returned")
    else:
        logging.error(f"{response.status} on returning {instance}")
        raise RuntimeError(f"Not ok response: {response.status}")
