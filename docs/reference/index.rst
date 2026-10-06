:description: API reference for the shared, dependency-light helper modules.
:group: Reference

API Reference
=============

``AIAgent/`` is an application (``package-mode = false`` in ``pyproject.toml``)
rather than an installable library, so most modules are scripts and are not
published as a stable public API. This page documents the shared, dependency-light
helper modules; add more modules to the autosummary list as they become
import-safe.

.. autosummary::
   :toctree: generated
   :nosignatures:

   common.typealias
   common.utils
   common.file_system_utils
