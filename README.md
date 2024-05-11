<p align="center">
  <img src="./resources/logo.png" width="256">
</p>

# PySymGym
Python infrastructure to train paths selectors for symbolic execution engines.


## Quick Start


This repository contains submodules, so use the following command to get sources locally. 
```sh
git clone https://github.com/gsvgit/PySymGym.git
git submodule update --init --recursive
```

Build .net game server (V#)
```sh
cd GameServers/VSharp
dotnet build -c Release
```

Create & activate virtual environment:
```bash
python3 -m pip install virtualenv
python3 -m virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

### GPU installation:

To use GPU, the correct `torch` and `torch_geometric` version should be installed depending on your host device. You may first need to `pip uninstall` these packages, provided by requirements.
Then follow installation instructions provided on [torch](https://pytorch.org/get-started/locally/) and [torch_geometric](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html#installation-from-wheels) websites.

## Usage

...

## Linting tools

Install [black](https://github.com/psf/black) code formatter by running following command in repo root to check your codestyle before committing:
```sh
pip install pre-commit && pre-commit install
```
