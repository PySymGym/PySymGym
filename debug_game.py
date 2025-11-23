import sys
import os
sys.path.insert(0, '/home/mihaelk/Desktop/code/PySymGym/AIAgent')

from AIAgent.game_environment import GameEnvironment
from AIAgent.config import Config
import logging

logging.basicConfig(level=logging.DEBUG)

# Загрузим конфиг
config_path = "/home/mihaelk/Desktop/code/PySymGym/workflow/config_edit_distance.yml"
config = Config.from_yaml(config_path)

# Создадим окружение
env = GameEnvironment(config)

# Запустим одну игру
print("=" * 80)
print("Starting game with EditDistance.GetMinDistance_BFS_0")
print("=" * 80)

result = env.play_game(
    map_name="EditDistance.GetMinDistance_BFS_0",
    steps=100
)

print("=" * 80)
print(f"Game result: {result}")
print("=" * 80)

# Посмотрим, что сохранилось
output_dir = "/home/mihaelk/Desktop/code/PySymGym/GameServers/VSharp/VSharp.ML.GameServer.Runner/bin/Release/net7.0/8100"
print(f"\nFiles in output directory:")
for root, dirs, files in os.walk(output_dir):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            size = os.path.getsize(filepath)
        except OSError:
            size = -1
        print(f"  {filepath}: {size} bytes")
