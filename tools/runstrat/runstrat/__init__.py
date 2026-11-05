"""The runstrat benchmarking tool.

Benchmarks a path-selection strategy against a set of .NET maps by driving
the V# test runner; see ``tools/README.md`` for usage. The public API is
re-exported here; the implementation lives in :mod:`runstrat.cli`.
"""

from runstrat.cli import (
    AllMethodsFailedError,
    Args,
    AssemblyInfo,
    DOTNET_VERSION,
    PATH_TO_VSHARP,
    RunMode,
    entrypoint,
    main,
    setup_logging,
)

__all__ = [
    "AllMethodsFailedError",
    "Args",
    "AssemblyInfo",
    "DOTNET_VERSION",
    "PATH_TO_VSHARP",
    "RunMode",
    "entrypoint",
    "main",
    "setup_logging",
]
