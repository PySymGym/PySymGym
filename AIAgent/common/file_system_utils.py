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
    empty_dir = Path("./empty_temp_dir_for_rsync")
    os.makedirs(empty_dir, exist_ok=True)
    try:
        _ = subprocess.run(
            ["rsync", "-a", "--delete", f"{empty_dir}/", f"{str(dir)}/"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(e, exc_info=True)
        raise
