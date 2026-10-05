"""Tests that keep the API reference in sync with the code it documents."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / "AIAgent" / "common"
REFERENCE = REPO_ROOT / "docs" / "reference" / "index.rst"

# Modules intentionally kept out of the API reference: importing them requires
# heavy or optional dependencies, so autodoc cannot safely process them. Every
# other module under AIAgent/common must be either documented in the reference
# or added here with a reason.
EXCLUDED = {"common.classes", "common.game", "common.network_utils"}


def _documented_modules() -> set[str]:
    modules: set[str] = set()
    in_block = False
    for line in REFERENCE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped.startswith(".. autosummary::"):
                in_block = True
            continue
        if not stripped:
            continue
        if not line.startswith(" "):
            break
        if stripped.startswith(":"):
            continue
        modules.add(stripped)
    return modules


def _actual_modules() -> set[str]:
    return {
        f"common.{path.stem}"
        for path in COMMON_DIR.glob("*.py")
        if path.stem != "__init__"
    }


def test_autosummary_references_only_existing_modules():
    stale = _documented_modules() - _actual_modules()
    assert not stale, f"docs/reference/index.rst lists missing modules: {sorted(stale)}"


def test_every_common_module_is_documented_or_excluded():
    untriaged = _actual_modules() - _documented_modules() - EXCLUDED
    assert not untriaged, (
        "AIAgent/common modules are neither in docs/reference/index.rst nor in "
        f"EXCLUDED: {sorted(untriaged)}"
    )
    contradictory = _documented_modules() & EXCLUDED
    assert not contradictory, (
        f"modules both documented and excluded: {sorted(contradictory)}"
    )


def test_documented_modules_have_numpydoc_docstrings():
    files = sorted(
        str(COMMON_DIR / f"{module.split('.')[-1]}.py")
        for module in _documented_modules()
    )
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "D", *files],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
