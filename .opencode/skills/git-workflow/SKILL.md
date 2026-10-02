---
name: git-workflow
description: Use when doing git operations: committing, branching, opening pull requests. Covers the operational procedure (commit validation, PR to main, pre-merge checks); the branching/commit model lives in docs/developer.rst.
---

# Git Workflow

The model (branching scheme, commit message format, merge strategy, quality
gate) is documented in the "Contribution guidelines" section of
`docs/developer.rst` (single source of truth). This skill keeps only the
operational procedure.

## Branches

Work on a feature branch created from `main`; never commit directly to `main`.
Integration happens exclusively through a pull request targeting `main`.

## Submodules

`GameServers/` and `maps/` contain git submodules (see `.gitmodules`). They are
external repositories:

- Never stage or commit changes inside a submodule path from this repo.
- After cloning or switching branches, initialize them with
  `git submodule update --init --recursive`.
- A changed submodule pointer is almost never part of a task's scope; if one
  appears in `git status`, stop and ask before touching it.

## Commits

**Pre-commit validation**: before running `git commit`, verify the prepared
message follows the format from the docs — Conventional Commits with exactly
one subtask identifier (`feat(XXX-SN): ...`; ranges, lists, or commas are
forbidden). If the message mentions multiple subtask identifiers, STOP —
split the changes into individual commits. One commit per completed atomic
subtask.

**Issue-closing validation**: every completed task carries the closing
keyword for its own issue (`Closes #<task issue>`) plus one per linked issue
it fully resolves (`Fixes #N` for defects, `Closes #N` otherwise) — all as
standalone lines in exactly one commit: the last subtask's commit. Verify on
the feature branch:

```bash
git log main..HEAD --format=%B | grep -cE '^(Fixes|Closes) #[0-9]+$'
```

must print `0` before the last subtask's commit and, afterwards, exactly one
line per fully resolved issue including the task's own (no more). A task that
only partially addresses a linked issue uses a bare `#N` reference (no
keyword). The model and closing timing live in the "Contribution guidelines"
section of `docs/developer.rst`.

### Pre-commit checklist

1. Run commit gate: code-style, all component tests, linters.

### Commit scope

- Each commit is a self-contained, compilable, testable increment
- Commit messages must be detailed enough to understand why changes were required

## Pull request to main

### Pre-merge checks

Run the quality gate (see the `quality-gates` skill). All checks MUST pass
before opening the PR. **This is absolute — no exceptions, no
self-assessment.**

### Procedure

Push the feature branch and open a PR into `main`. Rebase onto the latest
`main` first:

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease -u origin feature/XXX-short-description
gh pr create --base main --head feature/XXX-short-description \
  --title "<type>: <summary>" --body "<what/why, links to the task issue>"
```

The user reviews and merges the PR. Wait for their confirmation before
continuing. Merge with **rebase and merge** (fast-forward), not a merge commit
and not squash — this keeps `main` linear while preserving the per-subtask
commits the message format is built around.

After the merge, sync and clean up:

```bash
git checkout main
git pull
git branch -d feature/XXX-short-description
git push origin --delete feature/XXX-short-description
```

## Rules

- No emergency fixes
- Never force-push or rewrite `main` history; a feature branch may be
  force-pushed with `--force-with-lease` after a rebase
- Never merge your own PR without the user's confirmation
