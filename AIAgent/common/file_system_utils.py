import os
from pathlib import Path


def create_folders_if_necessary(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            os.makedirs(path)


def create_file(file: Path):
    open(file, "w").close()
