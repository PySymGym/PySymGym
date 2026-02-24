import json
import os
import pathlib
import re

import cattrs
import pandas as pd
from src.structs import LaunchInfo, PrebuiltConfig


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
            r"Tests generated: (?P<count>\d+)", runner_output
        ).groupdict()["count"]
        errs_generated = re.search(
            r"Errors generated: (?P<count>\d+)", runner_output
        ).groupdict()["count"]
        total_coverage = re.search(
            r"Precise coverage: (?P<count>.*)", runner_output
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

    for dll, method in zip(df["dll"], df["method"], strict=True):
        result.append(LaunchInfo(dll_dir / dll, method))

    return result
