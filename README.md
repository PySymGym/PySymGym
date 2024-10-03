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
pip install poetry
poetry install
```

### GPU installation:

To use GPU, the correct `torch` and `torch_geometric` version should be installed depending on your host device. You may first need to `pip uninstall` these packages, provided by requirements.
Then follow installation instructions provided on [torch](https://pytorch.org/get-started/locally/) and [torch_geometric](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html#installation-from-wheels) websites.

## Usage

### Run training
...

### ONNX conversion

To use ONNX conversion tool, locate `onyx.py` script in `AIAgent/` directory. Then run the following command:

```bash
onyx.py --sample-gamestate <game_state0.json> \
    --pytorch-model <model>.pt \
    --savepath <converted_model_save_path>.onnx \
    --import-model-fqn <model.module.fqn.Model> \
    --model-kwargs <yaml_with_model_args.yml> \
    [optional] --verify-on <game_state1.json> <game_state2.json> <game_state3.json> ...
```

model_kwargs yaml file, *verification* game_states and *sample* game_state (use any) can be found in [resources/onnx](resources/onnx/) dir

## Linting tools

Install [ruff](https://docs.astral.sh/ruff/) linter and code formatter by running following command in repo root to check your codestyle before committing:
```sh
pip install ruff

# to autofix all linting problems, run
ruff format
```

**Or** [integrate](https://docs.astral.sh/ruff/integrations/#vs-code-official) it with your favorite code editor (for example, [VSCode](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff))

Just to check CI.
