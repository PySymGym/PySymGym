<!--
The documentation contract is enforced by CI; see docs/developer.rst for the
full contribution guide and CONTRIBUTING.md for the pointers.
-->

## Summary

<!-- What changed and why. -->

## Checklist

- [ ] Code style passes: `ruff check` and `ruff format --check`.
- [ ] Tests pass for the affected component: `poetry run pytest tests -sv`.
- [ ] Documentation follows the project conventions (documented under `docs/`,
      not only `README.md`):
  - [ ] User-visible behavior and structure are documented.
  - [ ] New or changed pages carry `:description:` and `:group:` metadata.
  - [ ] `README.md` links still resolve.
- [ ] Docs build passes:
      `poetry run sphinx-build -W --keep-going -b html docs docs/_build/html`.

## Related issues

Closes #
