from pathlib import Path
import os


def read_configs(dir) -> list[Path]:
    return [
        Path(dir) / file
        for file in os.listdir(Path(dir))
        if file.endswith(".yml") or file.endswith(".yaml")
    ]
