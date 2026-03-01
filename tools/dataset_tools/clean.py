import sys
from pathlib import Path
import argparse

sys.path.append("../../AIAgent")

from common.game import GameMap


def get_bad_episodes(log_path):
    bad_episodes = []

    if not log_path.exists():
        return bad_episodes

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "immediate GameOver" in line:
            parts = line.split("on ")
            if len(parts) > 1:
                episode_name = parts[1].split()[0]
                if episode_name not in bad_episodes:
                    bad_episodes.append(episode_name)

    return bad_episodes


def clean(log_path, dataset_path):
    with open(dataset_path, "r") as f:
        dataset = GameMap.schema().loads(f.read(), many=True)

    bad_episodes = get_bad_episodes(log_path)
    result = []

    for game_map in dataset:
        if game_map.MapName not in bad_episodes:
            result.append(game_map)

    result.sort(
        key=lambda x: (x.NameOfObjectToCover, x.StepsToStart, x.DefaultSearcher)
    )

    with open(dataset_path, "w") as f:
        f.write(GameMap.schema().dumps(result, many=True, indent=4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=Path("../../maps/DotNet/Maps/dataset.json"),
        help="Path to the dataset JSON file",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=Path,
        default=Path("../../AIAgent/ml_app.log"),
        help="Path to the log file",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"Error: Dataset file not found at {args.dataset}")

    if not args.log.exists():
        sys.exit(f"Error: Log file not found at {args.log}")

    clean(args.log, args.dataset)


if __name__ == "__main__":
    main()
