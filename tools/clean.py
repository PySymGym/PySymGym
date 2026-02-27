import sys
from pathlib import Path
import argparse

sys.path.append("../AIAgent")

from common.game import GameMap


def get_maps(log_path):
    bad_maps = []

    if not log_path.exists():
        return bad_maps

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "immediate GameOver" in line:
            parts = line.split("on ")
            if len(parts) > 1:
                map_name = parts[1].split()[0]
                if map_name not in bad_maps:
                    bad_maps.append(map_name)

    return bad_maps


def clean(log_path, dataset_path):
    with open(dataset_path, "r") as f:
        dataset = GameMap.schema().loads(f.read(), many=True)

    bad_maps = get_maps(log_path)
    result = []

    for game_map in dataset:
        if game_map.MapName not in bad_maps:
            result.append(game_map)

    result.sort(
        key=lambda x: (x.NameOfObjectToCover, x.StepsToStart, x.DefaultSearcher)
    )
    with open(dataset_path, "w") as f:
        f.write(GameMap.schema().dumps(result, many=True, indent=3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=Path("../maps/DotNet/Maps/dataset.json"),
        help="Path to the dataset JSON file",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=Path,
        default=Path("../AIAgent/ml_app.log"),
        help="Path to the log file",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset file not found at {args.dataset}")
        sys.exit(1)

    if not args.log.exists():
        print(f"Error: Log file not found at {args.log}")
        sys.exit(1)

    clean(args.log, args.dataset)


if __name__ == "__main__":
    main()
