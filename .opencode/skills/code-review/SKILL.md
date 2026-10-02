---
name: code-review
description: Use after all subtasks are committed and before merging a task to main. Perform a whole-repo review of the task's changes, iterate to zero findings.
---

# Code Review

Run after all subtasks are committed, before the quality gate and merge.
Review the entire repo's diff for the task and iterate until there are zero
findings.

## Procedure

1. Determine the task's changes:
   `git diff --stat main...HEAD` and `git diff main...HEAD`.
2. Review against the checklist below.
3. For every finding, fix it in a follow-up commit on the same feature branch
   (one commit per subtask/fix as per `git-workflow`).
4. Repeat until zero findings.

## Checklist

- [ ] **Correctness** — every clause of the task issue (#N) is traceable to
      committed code.
- [ ] **Tests** — new code is covered; no test was weakened or skipped; the
      suite passes (see `run-tests`).
- [ ] **Docs** — docs updated per the `documentation` skill; docstrings follow
      its conventions.
- [ ] **Duplication** — no copy-pasted or near-identical logic (see `reusing`).
- [ ] **Style** — `code-style` pass clean; types consistent with the codebase.
- [ ] **Dead code** — no unused imports, functions, or debug artifacts.
- [ ] **Scope** — changes touch only what the task requires; no unrelated edits.

Zero findings means all boxes pass. Do not merge with known findings.
