"""Loading of the YAML training configs.

The path keys (``dataset_base_path``, ``dataset_description``,
``server_working_dir``) may be relative in the config file; they are resolved
against the config file's own directory — never the CWD — before the config
models see them (the models require absolute paths). The entrypoints load
their config through :func:`load_config`, which additionally validates that
every path names an existing filesystem object of the required kind, so a
broken config fails fast at load time instead of deep inside training.
"""

from pathlib import Path

import yaml

from common.config.config import Config

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


def validate_paths(data: object) -> list[str]:
    """Return one message per path key whose value does not name an existing object of the required kind.

    The structure is expected to be anchored already (absolute path values).
    Each message names the key, the resolved path, and the key's location in
    the config (e.g.
    ``ValidationConfig.validation_mode.PlatformsConfig[0].DatasetConfigs[0].dataset_base_path``).

    Parameters
    ----------
    data : object
        The anchored config structure.

    Returns
    -------
    list[str]
        One message per offending key; empty when every path is valid.
    """
    return _validate(data, "")


def _validate(value: object, location: str) -> list[str]:
    problems: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PATH_KEYS and isinstance(item, (str, Path)):
                problems.extend(_check_path(Path(item), key, location))
            else:
                problems.extend(
                    _validate(item, f"{location}.{key}" if location else key)
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            problems.extend(_validate(item, f"{location}[{index}]"))
    return problems


def _check_path(path: Path, key: str, location: str) -> list[str]:
    where = f"{location}.{key}" if location else key
    kind = PATH_KEYS[key]
    if not path.exists():
        return [f"{key}: {path} does not exist ({where})"]
    if kind == "directory" and not path.is_dir():
        return [f"{key}: {path} is not a directory ({where})"]
    if kind == "file" and not path.is_file():
        return [f"{key}: {path} is not a file ({where})"]
    return []


def load_config(config_path: str | Path) -> Config:
    """Load, anchor and validate a YAML training config.

    Relative path values are resolved against the config file's own
    directory (never the CWD); every path key must then name an existing
    filesystem object of the required kind, or a ``ValueError`` listing every
    offending key is raised before any training work starts.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file (resolved against the CWD, as CLI
        arguments are).

    Returns
    -------
    Config
        The validated config with absolute paths.

    Raises
    ------
    ValueError
        If a path key does not name an existing object of the required kind.
    """
    path = Path(config_path).expanduser().resolve()
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    anchored = anchor_paths(data, path.parent)
    problems = validate_paths(anchored)
    if problems:
        raise ValueError("config path validation failed:\n" + "\n".join(problems))
    return Config(**anchored)
