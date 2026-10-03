# Contributing

Thanks for contributing to PySymGym!

The contribution process — branching and integration model, commit message
format, tests, linting, and the docs build — is documented in the developer
guide:

- Rendered: <https://pysymgym.github.io/PySymGym/developer.html>
- Source: [`docs/developer.rst`](docs/developer.rst)

## Documentation is part of the change

Documentation is the map for this project. When your change affects behavior or
structure:

- Update the relevant page under `docs/`. New or changed pages start with
  `:description:` and `:group:` metadata; the navigation map is generated from
  that metadata.
- Keep `README.md` links pointing at pages that exist.
- Run the docs build (see the developer guide); it fails on missing metadata or
  broken README links.

The [pull request template](.github/pull_request_template.md) lists the checks.
