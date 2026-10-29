"""Unit tests for the runstrat output/config parsing."""

import json
from pathlib import Path

import pytest
from runstrat.parsing import (
    load_config,
    load_prebuilt_config,
    parse_prebuilt,
    parse_runner_output,
)
from runstrat.structs import PrebuiltConfig

pytestmark = pytest.mark.unit

RUNNER_OUTPUT = """Total time: 00:01:02.123
Tests generated: 5
Errors generated: 2
Precise coverage: 88.5
"""


def test_parse_runner_output_extracts_statistics() -> None:
    assert parse_runner_output(RUNNER_OUTPUT) == (62, 5, 2, 88.5)


def test_parse_runner_output_raises_with_note_on_invalid_input() -> None:
    with pytest.raises(AttributeError) as exc_info:
        parse_runner_output("not a runner output")

    assert "Parse failed on output" in "\n".join(exc_info.value.__notes__)


def test_parse_runner_output_accepts_single_digit_time_fields() -> None:
    output = (
        "Total time: 0:05:03.123.\n"
        "Tests generated: 1\n"
        "Errors generated: 0\n"
        "Precise coverage: 50\n"
    )
    assert parse_runner_output(output) == (303, 1, 0, 50.0)


def test_parse_runner_output_accepts_three_digit_hours() -> None:
    output = (
        "Total time: 100:00:00.000.\n"
        "Tests generated: 1\n"
        "Errors generated: 0\n"
        "Precise coverage: 50\n"
    )
    assert parse_runner_output(output) == (360000, 1, 0, 50.0)


def test_parse_runner_output_rejects_trailing_text_after_coverage() -> None:
    output = (
        "Total time: 00:01:02.123\n"
        "Tests generated: 5\n"
        "Errors generated: 2\n"
        "Precise coverage: 88.5 extra\n"
    )
    with pytest.raises(AttributeError) as exc_info:
        parse_runner_output(output)

    assert "Parse failed on output" in "\n".join(exc_info.value.__notes__)


def test_parse_prebuilt_expands_classes_and_top_level_functions() -> None:
    config = PrebuiltConfig(
        dll_dir="/dlls",
        dlls={"a.dll": {"Cls": ["M1", "M2"], "": ["Top"]}},
    )

    infos = parse_prebuilt(config)

    assert [(info.dll, info.method) for info in infos] == [
        ("/dlls/a.dll", "Cls.M1"),
        ("/dlls/a.dll", "Cls.M2"),
        ("/dlls/a.dll", "Top"),
    ]


def test_load_config_reads_dll_method_pairs(tmp_path: Path) -> None:
    csv_path = tmp_path / "maps.csv"
    csv_path.write_text("dll,method\nA.dll,Cls.M1\nB.dll,Top\n")

    infos = load_config(tmp_path / "dlls", csv_path)

    # ``load_config`` joins the dll dir onto each row, so ``dll`` is a ``Path``
    # (the ``LaunchInfo.dll`` annotation says ``str`` but the value is a Path).
    assert [(info.dll, info.method) for info in infos] == [
        (tmp_path / "dlls" / "A.dll", "Cls.M1"),
        (tmp_path / "dlls" / "B.dll", "Top"),
    ]


def test_load_prebuilt_config_structures_and_expands(tmp_path: Path) -> None:
    json_path = tmp_path / "config.json"
    json_path.write_text(
        json.dumps(
            {"dll_dir": "/dlls", "dlls": {"a.dll": {"Cls": ["M1", "M2"], "": ["Top"]}}}
        )
    )

    infos = load_prebuilt_config(str(json_path))

    assert [(info.dll, info.method) for info in infos] == [
        ("/dlls/a.dll", "Cls.M1"),
        ("/dlls/a.dll", "Cls.M2"),
        ("/dlls/a.dll", "Top"),
    ]
