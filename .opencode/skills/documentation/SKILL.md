---
name: documentation
description: Use when determining which docs to update for a code change. Maps source changes to required documentation actions and defines documentation completeness verification. The single source of truth for doc conventions in this project.
---

# Documentation

Sphinx docs live in `docs/` (sources) and are configured by `docs/conf.py`.
The project is an application (`package-mode = false`), not a published
library, so only dependency-light shared modules are included in the API
reference under `docs/reference/`; most modules are scripts. This skill is the
single source of truth for what docs to update when code changes.

## Mapping: source change -> doc action

| Source change | Required doc action |
|---|---|
| New dependency-light public function/class in `AIAgent/common/` | Add its module to the `autosummary` list in `docs/reference/index.rst`; write a numpydoc docstring |
| Changed public function | Update its numpydoc docstring (params, returns, examples) |
| New component or tool (e.g. under `AIAgent/`, `tools/`) | Add a section/entry to `README.md` and, if it is a top-level area, to the `docs/index.rst` toctree |
| New Sphinx page | Add a `*.rst` file under `docs/` and add it to the parent toctree |
| Removed/renamed public API | Update the `autosummary` list and any docstrings/links referencing it |
| User-visible behavior change | Update the relevant usage section of `README.md` |

## Docstring conventions

- numpydoc style (`Parameters`, `Returns`, `Examples`), parsed by
  `sphinx.ext.napoleon`.
- Keep `Examples` self-contained and output-stable.

## Completeness verification

A documentation update is complete when:

- [ ] At least one doc file (`docs/**` or `README.md`) was created or updated
      for the change.
- [ ] New shared public APIs appear in the correct `autosummary` list.
- [ ] Navigation (toctrees in `docs/index.rst` and sub-index pages) is updated
      for any new page.
- [ ] Docstrings follow numpydoc.

To build and check the docs, see the "Docs build" section of
`docs/developer.rst` (CI: `.github/workflows/docs.yaml`).
