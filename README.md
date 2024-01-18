# PySymGym
Python infrastructure to train paths selectors for symbolic execution engines.


## Launch guide

Firstly, follow the steps in [Environment setup](#environment-setup) to create virtual environment

...

## Contribution

Here's a quick guide to start contributing:

### Environment setup

In repository root run:
```bash
python3 -m pip install virtualenv
python3 -m virtualenv .env
source .env/bin/activate
pip install requirements.txt
```

### Linting

Install black formatter by running this command in repo root
```bash
pip install pre-commit && pre-commit install
```
to check your codestyle on pre-commit

