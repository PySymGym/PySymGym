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
documentation lives in `docs/` pages. The navigation map in `docs/index.rst`
is **generated** from per-page file-wide metadata: every page starts with

```rst
:description: One-line purpose of the page.
:group: <Overview|Guides|Development|Reference>
```

The `docs/_ext/navmap.py` Sphinx extension renders the map from that metadata
and injects the description as the page's opening line. The docs build
(`.github/workflows/docs.yaml`, warnings-as-errors) fails when a page lacks a
valid description/group or when `README.md` links to a non-existent page. This
skill is the single source of truth for *what* to update when code changes.

## Mapping: source change -> doc action

| Source change | Required doc action |
|---|---|
| New dependency-light public function/class in `AIAgent/common/` | Add its module to the `autosummary` list in `docs/reference/index.rst`; write a numpydoc docstring |
| Changed public function | Update its numpydoc docstring (params, returns, examples) |
| New component, tool, or workflow | Add or extend a page under `docs/` (`docs/architecture.rst` for structure; the matching guide otherwise) with `:description:`/`:group:` metadata |
| New Sphinx page | Add a `*.rst` file under `docs/` with `:description:`/`:group:` metadata and add it to the `docs/index.rst` hidden toctree |
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
- [ ] Every new/changed page carries valid `:description:`/`:group:` metadata
      (the map is generated from it) and is in the `docs/index.rst` toctree.
- [ ] Detailed behavior is documented in `docs/`, not re-added to `README.md`.
- [ ] Docstrings follow numpydoc.

For navigation conventions and how to improve them, see the
`project-navigation` skill. To build and check the docs, see the "Docs build"
section of `docs/developer.rst` (CI: `.github/workflows/docs.yaml`).
