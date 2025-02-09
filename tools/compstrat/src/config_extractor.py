import enum
import pathlib
import typing as t

import cattr
import yaml

from src.comparator import CompareConfig, DataSourceType


def _structure_config(config: dict[str, str]) -> CompareConfig:
    def structure_enum_by_value(
        enum_value: str, enum_type: enum.Enum
    ) -> DataSourceType:
        return enum_type(enum_value)

    converter = cattr.Converter()
    converter.register_structure_hook(DataSourceType, structure_enum_by_value)
    return converter.structure(config, CompareConfig)


def read_configs(configs_file_path: pathlib.Path) -> t.Sequence[CompareConfig]:
    with open(configs_file_path, "r") as config_yaml_file:
        configs = yaml.safe_load(config_yaml_file)

    return [_structure_config(it) for it in configs]
