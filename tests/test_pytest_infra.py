"""Smoke tests for the shared pytest infrastructure (task #558, Phase 0).

These guard the root configuration every component suite depends on: the marker
taxonomy is registered, the import mode is ``importlib`` (required so the three
same-named ``tests/test_pipeline.py`` modules can coexist), and every component
has a test directory wired into ``testpaths``.
"""

import ast
from pathlib import Path
from typing import cast

import pytest
from pytest import Config

pytestmark = pytest.mark.unit

TIERS = ("unit", "integration", "e2e")

UNIT_MARKERS = {
    "unit",
    "integration",
    "e2e",
    "gpu",
    "network",
    "slow",
    "serial",
}

COMPONENT_TEST_DIRS = [
    "AIAgent/tests",
    "tools/compstrat/tests",
    "tools/runstrat/tests",
    "tools/dataset_tools/tests",
    "tools/util/tests",
]

ROOT = Path(__file__).resolve().parent.parent


def test_marker_taxonomy_is_registered(pytestconfig: Config) -> None:
    markers = cast("list[str]", pytestconfig.getini("markers"))
    registered = {line.split(":", 1)[0].strip() for line in markers if line.strip()}
    assert UNIT_MARKERS <= registered


def test_import_mode_is_importlib(pytestconfig: Config) -> None:
    assert pytestconfig.getoption("importmode") == "importlib"


def test_every_component_test_dir_is_on_testpaths(pytestconfig: Config) -> None:
    configured = cast("list[str]", pytestconfig.getini("testpaths"))
    testpaths = {str(Path(path)) for path in configured}
    for component_dir in COMPONENT_TEST_DIRS:
        assert component_dir in testpaths


def test_every_component_contributes_at_least_one_test() -> None:
    for component_dir in COMPONENT_TEST_DIRS:
        directory = ROOT / component_dir
        test_files = list(directory.rglob("test_*.py"))
        assert test_files, f"{component_dir} has no test files"


def _tier_of_attribute(node: ast.AST) -> str | None:
    """Return the tier of a ``pytest.mark.<tier>`` attribute, else ``None``."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr in TIERS
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    ):
        return node.attr
    return None


def _tiers_in(value: ast.AST) -> set[str]:
    if isinstance(value, (ast.List, ast.Tuple)):
        return {
            tier
            for element in value.elts
            if (tier := _tier_of_attribute(element)) is not None
        }
    tier = _tier_of_attribute(value)
    return {tier} if tier is not None else set()


def _module_tiers(tree: ast.Module) -> set[str]:
    tiers: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if (
            any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in targets
            )
            and node.value is not None
        ):
            tiers |= _tiers_in(node.value)
    return tiers


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def _test_modules() -> list[Path]:
    modules: list[Path] = []
    for component_dir in [*COMPONENT_TEST_DIRS, "tests"]:
        modules.extend(sorted((ROOT / component_dir).rglob("test_*.py")))
    return modules


def test_every_test_has_exactly_one_tier_marker() -> None:
    """Every test resolves to exactly one tier, from its decorators or module."""
    problems: list[str] = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_tiers = _module_tiers(tree)
        functions = _test_functions(tree)
        if not functions:
            problems.append(f"{path}: no test functions")
            continue
        for func in functions:
            decorator_tiers = {
                tier
                for decorator in func.decorator_list
                if (tier := _tier_of_attribute(decorator)) is not None
            }
            tiers = decorator_tiers or module_tiers
            if len(tiers) != 1:
                label = sorted(tiers) if tiers else "no"
                problems.append(f"{path}:{func.lineno} {func.name} has {label} tier")
    assert not problems, "misclassified tests:\n" + "\n".join(problems)
