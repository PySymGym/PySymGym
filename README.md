<p align="center">
  <img src="./resources/logo.png" width="256">
</p>

# PySymGym
Python infrastructure to train paths selectors for symbolic execution engines.


## Install


This repository contains submodules, so use the following command to get sources locally. 
```sh
git clone https://github.com/gsvgit/PySymGym.git
git submodule update --init --recursive
```

Build Symbolic Virtual Machines ([V#](https://github.com/VSharp-team/VSharp) and [usvm](https://github.com/UnitTestBot/usvm)) and methods for training. To do this step you need dotnet7, cmake, clang and maven to be installed. 
```bash
cd ./PySymGym
make build_SVMs build_maps
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

### Generate initial dataset
To start supervised learning we need some initial data. It can be obtained using any path selection strategy. In our project we generate initial data with one of strategies from V#. To do it run:
```bash
make init_data
```
Now initial dataset saved in the directory `./AIAgent/report/SerializedEpisodes`. Then it will be updated by neural network if it finds a better solution.

### Run training
...


### ONNX conversion

To use ONNX conversion tool, locate `onyx.py` script in `AIAgent/` directory. Then run the following command:

```bash
python3 onyx.py --sample-gamestate <game_state0.json> \
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
