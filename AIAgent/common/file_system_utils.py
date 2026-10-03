"""Filesystem helpers used by the agent scripts."""

import logging
import os
import shutil
from pathlib import Path


def create_folders_if_necessary(paths: list[Path]) -> None:
    """Create each directory in ``paths`` if it does not already exist.

    Parameters
    ----------
    paths : list[Path]
        Directories to create.
    """
    for path in paths:
        if not path.exists():
            os.makedirs(path)


def create_file(file: Path):
    """Create an empty file at ``file``, truncating it if it exists.

    Parameters
    ----------
    file : Path
        File to create.
    """
    open(file, "w").close()


def delete_dir(dir: str | Path):
    """Delete the directory ``dir`` recursively.

    Parameters
    ----------
    dir : str | Path
        Directory to delete.
    """
    try:
        shutil.rmtree(dir)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise
