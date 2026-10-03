---
name: project-navigation
description: Use before exploring or searching the PySymGym codebase. Forces navigation through README.md and the docs map (docs/index.rst, docs/architecture.rst) instead of scanning the whole repository, and requires improving the documentation when navigation is slow or an area is undescribed.
---

# Project Navigation

Start every code exploration from the documentation. Do not scan the whole
repository with glob/grep to "find out how things work" — the documentation is
the map, and it must stay good enough to navigate by.

## Procedure

1. Read `README.md` — the entrypoint for navigating the project.
2. Read `docs/index.rst` — the navigation map of every documentation page.
3. Read `docs/architecture.rst` — the component map and the "where to look for
   X" table.
4. From the map, open only the component or guide you need, then read the
   specific source files it names.
5. Confirm with targeted searches (a named directory, a symbol); never
   enumerate the whole tree to build a mental model.

## Improve navigation as you go

Navigation is part of the product, not an afterthought. When the docs do not
answer a navigation question, or send you to the wrong place, fix it as part of
your task:

- Missing or unclear structure -> extend `docs/architecture.rst`.
- Undocumented workflow -> add or extend the relevant page under `docs/`.
- New page -> register it (and its brief description) in `docs/index.rst`.
- Follow the `documentation` skill for conventions and completeness.

## Rules

- Prefer the map, then targeted reads; "the code is the source of truth" does
  not mean "read all the code first".
- Keep this skill a thin pointer: do not restate README or docs content here.
