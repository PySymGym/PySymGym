---
name: workflow-management
description: Load first at the start of every session, before doing anything else. Drives the overall working loop for a task or set of tasks on an integration branch (main or a personal branch): global plan, one task at a time, feature branch, detailed plan, subtask execution, code review, quality gate, integration, final pull request.
---

# Workflow Management

The top-level loop that orchestrates all other workflow skills. For a
single task, load this skill first; it points to the skills that handle each
step.

## Core Rules

- Log all tasks as GitHub issues with the `task` label (`gh issue create
  --label task`). Use the description exactly as the user provided it — minimal
  changes, only splitting and numbering. The task ID is the issue number.
- Resolve the **integration branch** once per session:
  `INTEGRATION=$(git config --get pysymgym.integrationBranch || echo main)`.
  It is `main` in direct mode and a personal, long-lived branch in stacked
  mode. Never commit directly to the integration branch.
- Do tasks strictly **one at a time**. Each task gets its own feature branch
  (created from the integration branch), its own detailed plan, and its own
  integration step. Never combine multiple tasks in a single feature branch.
- In direct mode a task is integrated into `main` through its own pull
  request. In stacked mode a task is rebased and fast-forward merged into the
  integration branch and **no pull request is opened**; a single pull request
  into `main` is opened only on the user's explicit request.
- Each decision must be documented before implementation. Documentation must
  be detailed enough to reproduce the project from scratch and understand why
  each decision was made.
- Commit messages must be detailed enough to understand the reasons for
  changes.
- Documentation-only tasks (no `.py` files changed) skip code-specific gates
  (tests, lint, format, docs build) but still follow all other workflow rules:
  one task per branch, one commit per subtask, code review.

## Working Loop

0. If the user requests multiple tasks at once, first create one issue per
   task, then create a global plan in `tasks/global_plan.md` referencing the
   issue numbers (see the `planning` skill) before proceeding.
1. Ensure user-defined tasks, the global plan, and project architecture are
   aligned.
2. Choose exactly ONE open `task`-labeled issue that is not yet done: a task
   is done when its subtask commits are on the integration branch
   (`git log "$INTEGRATION" --format=%s | grep -cE '\(<N>-S[0-9]+\):'` > 0,
   where N is the issue number). List candidates with
   `gh issue list --label task --state open`.
3. Rebase the integration branch onto `origin/main` (task-boundary refresh, see
   `git-workflow`), then create a feature branch from the integration branch for
   this single task (branching model: the "Contribution guidelines" section of
   `docs/developer.rst`; procedure: `git-workflow`).
4. Generate a detailed plan in `tasks/detailed_plan.md`, decomposing the task
   into atomic subtasks (see `planning`), then post it as a comment on the
   task issue; the first line of the comment is the marker
   `<!-- detailed-plan -->`.
5. Load the `subtask-loop` skill, then execute each subtask using its cycle.
5a. Verify all subtasks are complete and unblocked. Check
    `tasks/detailed_plan.md`:
    - If any subtask is marked `[blocked]` or `[deferred]`, STOP immediately.
      The task is NOT complete. Do NOT proceed to code review or the PR.
      Report blocking subtasks to the user and await guidance.
    - If a subtask was attempted, not committed, and its work reverted, the
      subtask is NOT complete. Do not silently skip it.
5b. When continuing a partially-done task, analyze current state before any
    code changes:
    - Verify `git branch --show-current` is the correct feature branch.
    - Review committed subtasks: `git log --oneline` on the feature branch.
    - Read `tasks/detailed_plan.md` and `tasks/global_plan.md`.
    - Cross-reference committed file changes with planned subtasks:
      `git diff --stat HEAD.."$INTEGRATION"`.
    - Report status to the user: "S1-S3 committed, S4 pending, ..."
6. After all subtasks are done, perform code review on the entire repo (see
   the `code-review` skill). Iteratively detect and fix problems until zero
   findings.
7. Load the `quality-gates` skill and run the gate. It must show PASS. If
   BLOCKED, do not assess whether failures are pre-existing or unrelated to
   your changes; fix every failure and re-run until PASS.
8. Integrate the task (see `git-workflow`):
   - **Direct mode** (`$INTEGRATION` is `main`): push the feature branch and
     open a pull request into `main`, merged with rebase and merge
     (fast-forward) so the per-subtask commits are preserved and `main` stays
     linear. The user reviews and merges the PR manually; proceed only after
     they confirm the merge.
   - **Stacked mode** (`$INTEGRATION` is a personal branch): rebase and
     fast-forward merge the feature branch into `$INTEGRATION`, then delete the
     feature branch. Do **not** push for a PR and do **not** open a PR.
9. Verify the last subtask's commit carries `Closes #<N>` (the task's own
   issue) as a standalone line — the issue closes when the commit reaches
   `main`. See the Task Completeness Verification in the `subtask-loop` skill
   (the single source of truth for what "done" means).
10. Return to step 2 for the next task. In stacked mode, keep accumulating
    completed tasks on `$INTEGRATION`; open the request to `main` only via the
    finalize step below, on the user's explicit request.

## Finalize to main (stacked mode, explicit request only)

When — and only when — the user explicitly asks to open the request to `main`:

1. Rebase `$INTEGRATION` onto `origin/main` (see `git-workflow`) and re-run the
   aggregated code review and quality gate over the whole `main...integration`
   diff (see `code-review` and `quality-gates`).
2. Push `$INTEGRATION` and open one pull request with base `main` and head
   `$INTEGRATION` (see `git-workflow`).
3. The user reviews and merges the PR with rebase and merge. After the merge,
   sync `main`; never delete the long-lived integration branch.
