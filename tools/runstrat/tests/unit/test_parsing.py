"""Unit tests for the runstrat output/config parsing."""

import pytest
from src.parsing import parse_prebuilt, parse_runner_output
from src.structs import PrebuiltConfig

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
