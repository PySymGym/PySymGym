import json
import logging
from urllib.parse import urlencode

import httplib2
from config import WebsocketSourceLinks
from connection.broker_conn.classes import ServerInstanceInfo, SVMInfo


def acquire_instance(svm_info: SVMInfo) -> ServerInstanceInfo:
    response, content = httplib2.Http().request(
        WebsocketSourceLinks.GET_WS + "?" + urlencode(SVMInfo.to_dict(svm_info))
    )
    if response.status != 200:
        logging.error(f"{response.status} with {content=} on acquire_instance call")
        raise RuntimeError(f"Not ok response: {response}, {content}")
    acquired_instance = ServerInstanceInfo.from_json(
        json.loads(content.decode("utf-8"))
    )
    logging.info(f"acquired ws: {acquired_instance}")
    return acquired_instance


def return_instance(instance: ServerInstanceInfo):
    logging.info(f"returning: {instance}")

    response, content = httplib2.Http().request(
        WebsocketSourceLinks.POST_WS,
        method="POST",
        body=instance.to_json(),
    )

    if response.status == 200:
        logging.info(f"{instance} is returned")
    else:
        logging.error(f"{response.status} on returning {instance}")
        raise RuntimeError(f"Not ok response: {response.status}")
