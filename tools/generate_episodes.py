import json
from pathlib import Path

DATASET = Path("../maps/DotNet/Maps/dataset.json")

def generate_steps(steps_to_play):
    if steps_to_play <= 500:
        return list(range(50, steps_to_play, 50))
    elif steps_to_play <= 1000:
        return list(range(100, steps_to_play, 100))
    elif steps_to_play <= 10000:
        return list(range(200, steps_to_play, 200))
    else :
        return list(range(1000, steps_to_play, 1000))

def duplicate_checking(episodes, method, strategy, steps_to_start):
    for card in episodes:
        if (card.get("NameOfObjectToCover") == method and
            card.get("StepsToStart") == steps_to_start and
            card.get("DefaultSearcher") == strategy):
            return True
    return False

def main():
    with open(DATASET, 'r') as f:
        dataset = json.load(f)

    for item in dataset:
        method = item.get("NameOfObjectToCover")
        step = item.get("StepsToStart")
        strategy = item.get("DefaultSearcher")

        if step == 0:
            item["MapName"] = f"{method}_0"
        else:
            item["MapName"] = f"{method}_{step}_{strategy}"

    methods = {}
    for d in dataset:
        method = d.get("NameOfObjectToCover")
        if method not in methods:
            methods[method] = []
        methods[method].append(d)

    new_episodes = []

    for method, episodes in methods.items():
        if len(episodes):
            first_episode = episodes[0]
            steps_to_play = first_episode.get("StepsToPlay")
            steps = generate_steps(steps_to_play)

            for step in steps:
                for strategy in ["BFS", "DFS"]:
                    if not duplicate_checking(episodes, method, strategy, step):
                        new_episode = first_episode.copy()
                        new_episode["StepsToStart"] = step
                        new_episode["DefaultSearcher"] = strategy
                        new_episode["MapName"] = f"{method}_{step}_{strategy}"
                        new_episodes.append(new_episode)

            if not duplicate_checking(episodes, method, "BFS", 0):
                new_episode = first_episode.copy()
                new_episode["StepsToStart"] = 0
                new_episode["DefaultSearcher"] = "BFS"
                new_episode["MapName"] = f"{method}_0"
                new_episodes.append(new_episode)

    dataset.extend(new_episodes)
    dataset.sort(key=lambda x: (x.get("NameOfObjectToCover"), x.get("StepsToStart"), x.get("DefaultSearcher")))

    with open(DATASET, 'w') as f:
        json.dump(dataset, f, indent=3)

if __name__ == "__main__":
    main()
