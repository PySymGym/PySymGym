import json
import os
import pathlib
import re

import cattrs
import pandas as pd
from runstrat.structs import LaunchInfo, PrebuiltConfig


def _search_groupdict(pattern: str, text: str, *, flags: int = 0) -> dict[str, str]:
    """``re.search`` + ``groupdict``; a missing match is a parse failure.

    Raises the ``AttributeError`` the caller's ``except AttributeError``
    handler annotates with the offending output.
    """
    match = re.search(pattern, text, flags)
    if match is None:
        raise AttributeError(pattern)
    return match.groupdict()


def parse_runner_output(runner_output: str):
    try:
        total_time = _search_groupdict(
            r"Total time: (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)\.*",
            runner_output,
        )
        total_time = (
            int(total_time["hours"]) * 3600
            + int(total_time["minutes"]) * 60
            + int(total_time["seconds"])
        )
        test_generated = _search_groupdict(
            r"Tests generated: (?P<count>\d+)", runner_output
        )["count"]
        errs_generated = _search_groupdict(
            r"Errors generated: (?P<count>\d+)", runner_output
        )["count"]
        total_coverage = _search_groupdict(
            r"Precise coverage: (?P<count>\d+(?:\.\d+)?)\s*$",
            runner_output,
            flags=re.MULTILINE,
        )["count"]
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

    for dll, method in zip(df["dll"], df["method"], strict=True):
        result.append(LaunchInfo(dll_dir / dll, method))

    return result
