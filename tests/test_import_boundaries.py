"""Import boundary guard for the shared domain layer.

``AIAgent/common`` is the dependency-light layer everything else builds
on: importing it must never pull in the ML stack (``ml.*``) or a
top-level ``config`` module (the config system is the ``common.config``
package; a top-level ``config`` module would recreate the constant-class
system #585 merged into it). This guard pins that boundary statically,
following the AST-guard pattern of ``test_pytest_infra.py``: no module
under ``AIAgent/common`` may import ``ml.*`` or top-level ``config``
(``common.config`` is the config package and stays allowed).
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
COMMON_DIR = ROOT / "AIAgent" / "common"

# Top-level packages the shared layer must not import. ``common.config``
# starts with "common", not "config", so it is unaffected.
FORBIDDEN = ("ml", "config")


def _is_forbidden(module: str) -> bool:
    return any(module == name or module.startswith(name + ".") for name in FORBIDDEN)


def _violations(path: Path) -> list[str]:
    """Forbidden imports in one file, as ``file:line`` messages."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    problems.append(f"{path}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _is_forbidden(node.module):
                problems.append(f"{path}:{node.lineno} from {node.module} import ...")
    return problems


def test_common_does_not_import_ml_or_top_level_config() -> None:
    problems: list[str] = []
    for path in sorted(COMMON_DIR.rglob("*.py")):
        problems.extend(_violations(path.relative_to(ROOT)))
    assert not problems, "import boundary violations:\n" + "\n".join(problems)
