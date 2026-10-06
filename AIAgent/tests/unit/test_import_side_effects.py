"""Importing the runnable AIAgent scripts must not touch the filesystem.

Logging configuration, directory creation and similar side effects belong to
``main()`` (or the ``if __name__ == "__main__"`` guard). Importing a module in
a test, a documentation build or a tool must never write to the current
working directory. These tests re-execute each script's module body inside an
empty temporary working directory and assert that nothing was created.
"""

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

MODULES = ["run_training", "launch_servers", "pretrain"]


@contextmanager
def _reimport(module_name: str):
    """Re-execute ``module_name``'s module body and restore ``sys.modules``.

    The caller must have imported the module (and thus its dependency graph)
    beforehand so that only the target module's top-level statements run.
    """
    saved = sys.modules.pop(module_name)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        sys.modules[module_name] = saved


@pytest.mark.parametrize("module_name", MODULES)
def test_import_creates_no_files(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importlib.import_module(module_name)
    monkeypatch.chdir(tmp_path)

    with _reimport(module_name):
        pass

    assert list(tmp_path.iterdir()) == []
