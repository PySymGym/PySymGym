---
name: git-workflow
description: Use when doing git operations: committing, branching, integrating a task, opening pull requests. Covers the operational procedure (commit validation, integration into the integration branch, final PR to main, pre-merge checks); the branching/commit model lives in docs/developer.rst.
---

# Git Workflow

The model (branching scheme, commit message format, merge strategy, quality
gate) is documented in the "Contribution guidelines" section of
`docs/developer.rst` (single source of truth). This skill keeps only the
operational procedure.

## Branches

All work targets an **integration branch**, resolved once per session:

```bash
INTEGRATION=$(git config --get pysymgym.integrationBranch || echo main)
```

It is `main` by default (direct mode). A developer may redirect it to a
personal, long-lived branch (stacked mode) with
`git config pysymgym.integrationBranch <branch>`. Work on a feature branch
created from the integration branch; never commit directly to the integration
branch. Integration into `main` is exclusively through a pull request;
integration into a personal branch is by rebase + fast-forward merge (see
below). The model lives in the "Contribution guidelines" section of
`docs/developer.rst`.

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
git log "$INTEGRATION"..HEAD --format=%B | grep -cE '^(Fixes|Closes) #[0-9]+$'
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

## Integrate a task

### Fast-forward into a personal integration branch (stacked mode)

When `$INTEGRATION` is not `main`, integrate the finished task by
fast-forward merging its feature branch into the integration branch:

```bash
git checkout "$INTEGRATION"
git merge --ff-only feature/XXX-short-description
git branch -d feature/XXX-short-description
```

If the feature branch is not a descendant of the integration branch (it moved),
rebase first, re-run the quality gate, then merge:

```bash
git checkout feature/XXX-short-description
git rebase "$INTEGRATION"
git checkout "$INTEGRATION"
git merge --ff-only feature/XXX-short-description
git branch -d feature/XXX-short-description
```

No pull request is opened in stacked mode. Repeat for the next task.

### Periodic rebase of the long-lived integration branch

A personal integration branch is never deleted. Keep it close to `main` by
rebasing onto the latest `main` at task boundaries and immediately before the
final pull request, then force-pushing with lease:

```bash
git checkout "$INTEGRATION"
git fetch origin
git rebase origin/main
git push --force-with-lease
```

## Final pull request to main (explicit request only)

In direct mode (`$INTEGRATION` is `main`) one pull request is opened per task.
In stacked mode open a pull request **only on the user's explicit request**,
after several tasks have accumulated on the integration branch.

### Pre-merge checks

Run the quality gate (see the `quality-gates` skill). In stacked mode also run
the aggregated review and gate over the whole `main...integration` diff. All
checks MUST pass before opening the PR. **This is absolute — no exceptions, no
self-assessment.**

### Procedure

Rebase the integration branch onto the latest `main` and open the PR:

```bash
git checkout "$INTEGRATION"
git fetch origin
git rebase origin/main
git push --force-with-lease -u origin "$INTEGRATION"
gh pr create --base main --head "$INTEGRATION" \
  --title "<type>: <summary>" --body "<what/why, links to the task issues>"
```

For direct mode, push the feature branch and use it as the PR head with base
`main` instead.

The user reviews and merges the PR. Wait for their confirmation before
continuing. Merge with **rebase and merge** (fast-forward), not a merge commit
and not squash — this keeps `main` linear while preserving the per-subtask
commits the message format is built around.

After the merge, sync `main`. Delete a feature branch once merged; never delete
the long-lived integration branch:

```bash
git checkout main
git pull
git branch -d feature/XXX-short-description
git push origin --delete feature/XXX-short-description
```

## Rules

- No emergency fixes
- Never force-push or rewrite `main` history; a feature branch or a personal
  integration branch may be force-pushed with `--force-with-lease` after a
  rebase
- Never open a PR in stacked mode, and never push for a PR, without the user's
  explicit request
- Never delete the long-lived integration branch
- Never merge your own PR without the user's confirmation
