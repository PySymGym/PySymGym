"""Small helpers shared across the agent packages."""

from pathlib import Path


def require_absolute_path(value: str | Path) -> Path:
    """Return ``value`` as an absolute :class:`~pathlib.Path`.

    Relative paths are rejected: they are ambiguous without an anchor, and
    the config loader resolves them against the config file's location
    before the config models see them.

    Parameters
    ----------
    value : str | Path
        The path to check.

    Returns
    -------
    Path
        The same path, as an absolute Path.

    Raises
    ------
    ValueError
        If ``value`` is not absolute.
    """
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(
            f"{value!r} is not an absolute path; relative config paths are "
            "resolved against the config file's location at load time"
        )
    return path
