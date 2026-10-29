"""The compstrat strategy-comparison tool.

Compares the results of two path-selection strategies over the same maps and
renders the comparison; see ``tools/README.md`` for usage. The public API is
re-exported here; the implementation lives in :mod:`compstrat.cli`.
"""

from compstrat.cli import Args, entrypoint, main

__all__ = [
    "Args",
    "entrypoint",
    "main",
]
