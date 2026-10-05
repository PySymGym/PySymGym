"""Root pytest configuration shared by every component suite.

The repository is a single Poetry project with per-component test trees. This
conftest makes their modules importable regardless of the current working
directory by putting each component directory on ``sys.path``; the project's
modules are imported by their top-level names (``common.*``, ``ml.*``,
``config``, ``compstrat``, ``runstrat``, ``src.*``).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent

COMPONENT_DIRS = [
    ROOT / "AIAgent",
    ROOT / "tools" / "compstrat",
    ROOT / "tools" / "runstrat",
    ROOT / "tools" / "dataset_tools",
    ROOT / "tools" / "util",
]

for component_dir in reversed(COMPONENT_DIRS):
    path = str(component_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
