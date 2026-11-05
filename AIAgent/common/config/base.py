"""Shared pydantic configuration for the config models.

Every pydantic config model in the project is built on :data:`CONFIG` so
the contract has a single source of truth. The two keys:

- ``extra="forbid"`` — unknown keys are rejected, so a typo in a YAML
  config fails fast at load time instead of being silently ignored.
- ``populate_by_name=True`` — fields may be populated by their Python name
  as well as their alias: the code constructs the models with snake_case
  kwargs, the YAML configs use the PascalCase aliases.

A model that needs a different behaviour overrides the base dict
(``{**CONFIG, ...}``) instead of redefining the contract.
"""

from pydantic import ConfigDict

#: The shared pydantic config of every config model (see the module
#: docstring).
CONFIG = ConfigDict(extra="forbid", populate_by_name=True)
