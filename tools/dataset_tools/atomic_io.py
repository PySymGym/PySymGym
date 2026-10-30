"""Atomic file replacement for the dataset tools.

The dataset tools rewrite the curated training dataset in place. Writing
through a temporary file in the same directory and renaming it over the
target (``os.replace``) makes the rewrite atomic: a crash or
``KeyboardInterrupt`` mid-write leaves the previous dataset intact instead
of a truncated file.
"""

import contextlib
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace the contents of ``path`` with ``text``.

    The text is written to a temporary file in the same directory as
    ``path`` (so the rename stays on one filesystem), flushed and fsynced,
    then renamed over ``path`` with ``os.replace``. On any failure —
    including ``KeyboardInterrupt`` — the temporary file is removed and the
    exception re-raised, leaving ``path`` untouched.

    Parameters
    ----------
    path : Path
        The target file to replace.
    text : str
        The new contents, written as UTF-8.

    Examples
    --------
    >>> from pathlib import Path
    >>> atomic_write_text(Path("dataset.json"), '[{"MapName": "A_0"}]')
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # The temp file is gone if the rename already succeeded or another
        # process cleaned it up; never mask the original exception.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
