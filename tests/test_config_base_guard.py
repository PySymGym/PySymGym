"""Static guard: every pydantic config model shares the base config.

The config models' pydantic config has one source of truth,
``common.config.base.CONFIG`` (the strict, alias-or-name contract). This
guard pins that statically, following the AST-guard pattern of
``test_import_boundaries.py`` (an AST walk that never imports the guarded
code): every ``@pydantic_dataclass`` decorator under ``AIAgent/common``
must take its config from the shared base — ``config=CONFIG`` imported from
``common.config.base``, or an explicit override that spreads it
(``config={**CONFIG, ...}``). A module-local copy of the dict (even an
identical one) would drift from the shared contract and is rejected here;
a new override is a conscious decision, so it must be visible in the diff.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
COMMON_DIR = ROOT / "AIAgent" / "common"


def _is_shared_config_ref(node: ast.AST) -> bool:
    """True when ``node`` is the shared ``CONFIG`` or an override spreading it."""
    if isinstance(node, ast.Name) and node.id == "CONFIG":
        return True
    return isinstance(node, ast.Dict) and any(
        key is None and isinstance(value, ast.Name) and value.id == "CONFIG"
        for key, value in zip(node.keys, node.values, strict=True)
    )


def _violations(path: Path) -> list[str]:
    """Shared-config violations in one file, as ``file:line`` messages."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports_shared_config = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "common.config.base"
        and any(alias.name == "CONFIG" for alias in node.names)
        for node in ast.walk(tree)
    )
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "pydantic_dataclass":
                # A bare decorator cannot carry a config at all.
                problems.append(
                    f"{path}:{decorator.lineno} pydantic_dataclass without config"
                )
            elif (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "pydantic_dataclass"
            ):
                config_kwarg = next(
                    (kw for kw in decorator.keywords if kw.arg == "config"), None
                )
                if config_kwarg is None:
                    problems.append(
                        f"{path}:{decorator.lineno} pydantic_dataclass without config"
                    )
                elif not _is_shared_config_ref(config_kwarg.value):
                    problems.append(
                        f"{path}:{decorator.lineno} pydantic_dataclass config "
                        "does not derive from the shared CONFIG"
                    )
                elif not imports_shared_config:
                    problems.append(
                        f"{path}:{decorator.lineno} CONFIG is not imported from "
                        "common.config.base"
                    )
    return problems


def test_every_config_model_uses_the_shared_base_config() -> None:
    problems: list[str] = []
    for path in sorted(COMMON_DIR.rglob("*.py")):
        problems.extend(_violations(path.relative_to(ROOT)))
    assert not problems, "shared config violations:\n" + "\n".join(problems)
