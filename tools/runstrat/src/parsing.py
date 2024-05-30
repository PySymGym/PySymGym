import json
import os
import pathlib
import re
import typing as t
from itertools import chain

import cattrs
import pandas as pd

from src.structs import LaunchInfo, PrebuiltConfig


def parse_maps(maps_file_path: pathlib.Path) -> t.Iterable[LaunchInfo]:
    with open(maps_file_path, "r") as maps_file:
        code_lines = maps_file.readlines()

    def _h1(s: list[str]):
        return LaunchInfo(s[2].strip('"'), s[4].strip('"'), None)

    test_maps = map(
        lambda s: _h1(s.strip().strip("\n").split(" ")),
        filter(
            lambda s: re.search("add \du<percent> ", s) != None and "//" not in s,
            code_lines,
        ),
    )

    def _h2(s: list[str]):
        return LaunchInfo(
            s[3].strip('"'), s[5].strip('"'), re.search("\d+", s[1]).group()
        )

    validation_maps = map(
        lambda s: _h2(s.strip().strip("\n").split(" ")),
        filter(
            lambda s: re.search("add \d+u<step> ", s) != None and "//" not in s,
            code_lines,
        ),
    )

    return chain(test_maps, validation_maps)


def dll_prepend(
    dll_dir: str, dll2methods: t.Iterable[LaunchInfo]
) -> t.Iterable[LaunchInfo]:
    def prepend2first(entry: LaunchInfo):
        return LaunchInfo(os.path.join(dll_dir, entry.dll), entry.method, entry.steps)

    return map(prepend2first, dll2methods)


def recurse_find(key: list[str], dotcover_out: dict) -> None | dict:
    if len(key) == 0:
        return dotcover_out

    for value in dotcover_out.values():
        if isinstance(value, str) and key[0] in value:
            if "Children" in dotcover_out:
                return recurse_find(key[1:], dotcover_out)
            return dotcover_out

    if "Children" not in dotcover_out.keys():
        return None

    for child in dotcover_out["Children"]:
        found = recurse_find(key, child)
        if found != None:
            return found

    return None


def parse_runner_output(runner_output: str):
    try:
        total_time = re.search(
            r"Total time: (?P<hours>\d\d):(?P<minutes>\d\d):(?P<seconds>\d\d)\.*",
            runner_output,
        ).groupdict()
        total_time = (
            int(total_time["hours"]) * 3600
            + int(total_time["minutes"]) * 60
            + int(total_time["seconds"])
        )
        test_generated = re.search(
            "Tests generated: (?P<count>\d+)", runner_output
        ).groupdict()["count"]
        errs_generated = re.search(
            "Errors generated: (?P<count>\d+)", runner_output
        ).groupdict()["count"]
        total_coverage = re.search(
            "Precise coverage: (?P<count>.*)", runner_output
        ).groupdict()["count"]
    except AttributeError as e:
        e.add_note(f"Parse failed on output:\n{runner_output}")
        raise

    return (
        total_time,
        int(test_generated),
        int(errs_generated),
        float(total_coverage),
    )


def parse_prebuilt(config: PrebuiltConfig) -> list[LaunchInfo]:
    launch_infos = []

    for dll, class2methods in config.dlls.items():
        for clazz, methods in class2methods.items():
            for method in methods:
                launch_infos.append(
                    LaunchInfo(
                        dll=os.path.join(config.dll_dir, dll),
                        method=method if clazz == "" else ".".join([clazz, method]),
                    )
                )

    return launch_infos


def load_prebuilt_config(config_json_path: str):
    with open(config_json_path, "r") as config_file:
        return parse_prebuilt(
            config=cattrs.structure(json.load(config_file), PrebuiltConfig)
        )


def load_config(dll_dir: pathlib.Path, config_csv_path: pathlib.Path):
    df = pd.read_csv(config_csv_path)

    result = []

    for dll, method in zip(df["dll"], df["method"]):
        result.append(LaunchInfo(dll_dir / dll, method))

    return result
