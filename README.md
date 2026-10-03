<p align="center">
  <img src="./resources/logo.png" width="256">
</p>

# PySymGym

[![Python linting](https://github.com/PySymGym/PySymGym/actions/workflows/python_linting.yaml/badge.svg)](https://github.com/PySymGym/PySymGym/actions/workflows/python_linting.yaml)
[![Build SVM-s and maps, and run training](https://github.com/PySymGym/PySymGym/actions/workflows/build_and_run.yaml/badge.svg)](https://github.com/PySymGym/PySymGym/actions/workflows/build_and_run.yaml)
[![Run python tests](https://github.com/PySymGym/PySymGym/actions/workflows/python_tests.yaml/badge.svg)](https://github.com/PySymGym/PySymGym/actions/workflows/python_tests.yaml)

Python infrastructure to train path selectors for symbolic execution engines.

## Install

This repository contains submodules, so use the following command to get sources locally.

```sh
git clone https://github.com/PySymGym/PySymGym.git
git submodule update --init --recursive
```

Setup environment:

```bash
pip install poetry==2.2.1
poetry install
```

### GPU installation:

To use GPU, the correct `torch` and `torch_geometric` version should be installed depending on your host device. You may first need to `pip uninstall` these packages, provided by requirements.
Then follow installation instructions provided on [torch](https://pytorch.org/get-started/locally/) and [torch_geometric](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html#installation-from-wheels) websites.

## Documentation

The documentation is the map for navigating the project. Start at the
[docs home](https://pysymgym.github.io/PySymGym/), or jump directly to a page:

- [Architecture](https://pysymgym.github.io/PySymGym/architecture.html)
- [Usage](https://pysymgym.github.io/PySymGym/usage.html)
- [Integration](https://pysymgym.github.io/PySymGym/integration.html)
- [Dataset](https://pysymgym.github.io/PySymGym/dataset.html)
- [Results](https://pysymgym.github.io/PySymGym/results.html)
- [Developer guide](https://pysymgym.github.io/PySymGym/developer.html)
- [API reference](https://pysymgym.github.io/PySymGym/reference/)
