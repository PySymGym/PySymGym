import sys
from pathlib import Path
from collections import defaultdict
import argparse

sys.path.append("../AIAgent")

from common.game import GameMap


def generate_steps(steps_to_play):
    if steps_to_play <= 500:
        return list(range(50, steps_to_play, 50))
    elif steps_to_play <= 1000:
        return list(range(100, steps_to_play, 100))
    elif steps_to_play <= 10000:
        return list(range(200, steps_to_play, 200))
    else:
        return list(range(1000, steps_to_play, 1000))


def is_duplicate(episodes, method, strategy, steps_to_start):
    for game_map in episodes:
        if (game_map.NameOfObjectToCover == method and
            game_map.StepsToStart == steps_to_start and
            game_map.DefaultSearcher == strategy):
            return True
    return False


def create_episode(base_episode, method, step, strategy):
    if step == 0:
        map_name = f"{method}_0"
    else:
        map_name = f"{method}_{step}_{strategy}"

    return GameMap(
        StepsToPlay=base_episode.StepsToPlay,
        DefaultSearcher=strategy,
        StepsToStart=step,
        AssemblyFullName=base_episode.AssemblyFullName,
        NameOfObjectToCover=method,
        MapName=map_name
    )


def generate_episodes(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = GameMap.schema().loads(f.read(), many=True)

    methods_dict = defaultdict(list)
    for episode in dataset:
        method = episode.NameOfObjectToCover
        methods_dict[method].append(episode)

    new_episodes = []
    for method, episodes in methods_dict.items():
        if len(episodes) > 0:
            first_episode = episodes[0]
            steps_to_play = first_episode.StepsToPlay
            steps = generate_steps(steps_to_play)

            for step in steps:
                for strategy in ["BFS", "DFS"]:
                    if not is_duplicate(episodes, method, strategy, step):
                        new_episode = create_episode(first_episode, method, step, strategy)
                        new_episodes.append(new_episode)

            if not is_duplicate(episodes, method, "BFS", 0):
                new_episode = create_episode(first_episode, method, 0, "BFS")
                new_episodes.append(new_episode)

    dataset.extend(new_episodes)
    dataset.sort(key=lambda x: (x.NameOfObjectToCover, x.StepsToStart, x.DefaultSearcher))

    with open(dataset_path, 'w') as f:
        f.write(GameMap.schema().dumps(dataset, many=True, indent=3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=Path("../maps/DotNet/Maps/dataset.json"),
        help="Path to the dataset JSON file"
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset file not found at {args.dataset}")
        sys.exit(1)

    generate_episodes(args.dataset)


if __name__ == "__main__":
    main()
