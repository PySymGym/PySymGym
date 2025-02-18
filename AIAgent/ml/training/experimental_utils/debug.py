import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from common.game import GameMap


def find_unfinished_maps(log_file_path: Path) -> None:
    server_log = open(log_file_path)
    ports = defaultdict(list)
    for line in reversed(list(server_log)):
        splitted = line.split(" ")
        status, map_name, port = splitted[0], splitted[2], splitted[4]
        if status == "Finish":
            ports[port].append(map_name)
        if status == "Start":
            try:
                ports[port].remove(map_name)
            except ValueError:
                print(map_name)
    server_log.close()


def sync_dataset_with_description(dataset_path: Path, description_path: Path) -> None:
    with open(description_path, "r") as maps_json:
        maps_in_description = list(
            map(
                lambda game_map: game_map.MapName,
                GameMap.schema().load(json.loads(maps_json.read()), many=True),
            )
        )
        for saved_map in os.listdir(dataset_path):
            if saved_map not in maps_in_description:
                shutil.rmtree(os.path.join(dataset_path, saved_map))
