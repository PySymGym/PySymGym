import logging
import os
import subprocess
from pathlib import Path


def create_folders_if_necessary(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            os.makedirs(path)


def create_file(file: Path):
    open(file, "w").close()


def delete_dir(dir: str | Path):
    try:
        os.rmdir(str(dir))
    except subprocess.CalledProcessError as e:
        logging.error(e, exc_info=True)
        raise
