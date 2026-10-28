"""Anchoring of the path values carried by a YAML training config.

The path keys (``dataset_base_path``, ``dataset_description``,
``server_working_dir``) may be relative in the config file; they are resolved
against the config file's own directory — never the CWD — before the config
models see them (the models require absolute paths).
"""

from pathlib import Path

# The path-valued config keys and the kind of filesystem object each must
# name. Every occurrence is anchored and validated, wherever it appears in
# the config (direct validation modes and custom ``val_sequence`` alike).
PATH_KEYS: dict[str, str] = {
    "dataset_base_path": "directory",
    "dataset_description": "file",
    "server_working_dir": "directory",
}


def anchor_paths(data: object, anchor: Path) -> object:
    """Return a copy of ``data`` with relative path values resolved against ``anchor``.

    The walk is key-based: only values under a :data:`PATH_KEYS` name are
    touched, every other value (including other strings) passes through
    unchanged. Absolute path values are kept as-is.

    Parameters
    ----------
    data : object
        The raw config structure (as loaded from YAML).
    anchor : Path
        The directory relative path values are resolved against (the config
        file's own directory).

    Returns
    -------
    object
        A new structure with the path values anchored; ``data`` is not
        mutated.
    """
    return _anchor(data, anchor)


def _anchor(value: object, anchor: Path) -> object:
    if isinstance(value, dict):
        anchored = {}
        for key, item in value.items():
            if key in PATH_KEYS and isinstance(item, (str, Path)):
                path = Path(item)
                anchored[key] = str(
                    path if path.is_absolute() else (anchor / path).resolve()
                )
            else:
                anchored[key] = _anchor(item, anchor)
        return anchored
    if isinstance(value, list):
        return [_anchor(item, anchor) for item in value]
    return value
