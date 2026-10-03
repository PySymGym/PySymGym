---
name: documentation
description: Use when determining which docs to update for a code change. Maps source changes to required documentation actions and defines documentation completeness verification. The single source of truth for doc conventions in this project.
---

# Documentation

Sphinx docs live in `docs/` (sources) and are configured by `docs/conf.py`.
The project is an application (`package-mode = false`), not a published
library, so only dependency-light shared modules are included in the API
reference under `docs/reference/`; most modules are scripts.

`README.md` is the entrypoint for navigating the project; the detailed
documentation lives in `docs/` pages. The navigation structure and the
"brief description at start" convention are expressed in `docs/index.rst`
(the navigation map) — this skill is the single source of truth for *what* to
update when code changes, and points there for *where* the map lives.

## Mapping: source change -> doc action

| Source change | Required doc action |
|---|---|
| New dependency-light public function/class in `AIAgent/common/` | Add its module to the `autosummary` list in `docs/reference/index.rst`; write a numpydoc docstring |
| Changed public function | Update its numpydoc docstring (params, returns, examples) |
| New component, tool, or workflow | Add or extend a page under `docs/` (`docs/architecture.rst` for structure; the matching guide otherwise) and register it in `docs/index.rst` (map + toctree) with a brief description |
| New Sphinx page | Add a `*.rst` file under `docs/`, open it with a brief one-line description, and register it in the `docs/index.rst` map and toctree |
| Removed/renamed public API | Update the `autosummary` list and any docstrings/links referencing it |
| User-visible behavior change | Update the relevant page under `docs/` (not `README.md`) |
| Change to the entrypoint/hub itself (install, top-level nav links, one-line description) | Update `README.md` |

## Docstring conventions

- numpydoc style (`Parameters`, `Returns`, `Examples`), parsed by
  `sphinx.ext.napoleon`.
- Keep `Examples` self-contained and output-stable.

## Completeness verification

A documentation update is complete when:

- [ ] At least one doc file (`docs/**` or `README.md`) was created or updated
      for the change.
- [ ] New shared public APIs appear in the correct `autosummary` list.
- [ ] Every new page is registered in `docs/index.rst` with a brief
      description, and the map/toctree navigation is updated.
- [ ] Detailed behavior is documented in `docs/`, not re-added to `README.md`.
- [ ] Docstrings follow numpydoc.

For navigation conventions and how to improve them, see the
`project-navigation` skill. To build and check the docs, see the "Docs build"
section of `docs/developer.rst` (CI: `.github/workflows/docs.yaml`).
