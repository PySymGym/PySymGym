# Usage

Prerequisites: cd into tools' directory

## `runstrat` tool

Docs:

```bash
usage: runstrat.py [-h] -s STRATEGY [-mp MODEL_PATH] -t TIMEOUT [-ps PYSYMGYM_PATH] [-as dlls-path launch-info-path]

options:
  -h, --help            show this help message and exit
  -s STRATEGY, --strategy STRATEGY
                        V# searcher strategy
  -mp MODEL_PATH, --model-path MODEL_PATH
                        Absolute path to AI model if AI strategy is selected
  -t TIMEOUT, --timeout TIMEOUT
                        V# runner timeout
  -ps PYSYMGYM_PATH, --pysymgym-path PYSYMGYM_PATH
                        Absolute path to PySymGym
  -as dlls-path launch-info-path, --assembly-infos dlls-path launch-info-path
                        Provide tuples: dir with dlls/assembly info file
```

To start benchmark, run

```bash
python3 runstrat.py \
    --strategy AI \
    --model-path <..>/model.onnx \
    --timeout 120 \
    --pysymgym-path <abs-path-to-PySymGym> \
    --assembly-infos <abs-path-to-PySymGym>/maps/DotNet/Maps/Root/bin/Release/net7.0 <abs-path-to-prebuilt>/assembled.csv \
    --assembly-infos <abs-path-to-prebuilt>/cosmos/publish prebuilt/cosmos_os.csv \
    --assembly-infos <abs-path-to-prebuilt>/powershell-osx-arm64 prebuilt/powershell.csv
```

Or shortened version:

```bash
python3 runstrat.py \
    -s AI \
    -t 120 \
    -ps <abs-path-to-PySymGym> \
    -as <abs-path-to-PySymGym>/maps/DotNet/Maps/Root/bin/Release/net7.0 <path-to-prebuilt>/assembled.csv \
    -as <path-to-prebuilt>/cosmos/publish prebuilt/cosmos_os.csv \
    -as <path-to-prebuilt>/powershell-osx-arm64 prebuilt/powershell.csv
```

## `compstrat` tool

Docs:
```bash
usage: compstrat.py [-h] -s1 STRAT1 -r1 RUNS1 [RUNS1 ...] -s2 STRAT2 -r2 RUNS2 [RUNS2 ...] -cp CONFIGS_PATH [--savedir SAVEDIR]

options:
  -h, --help            show this help message and exit
  -s1 STRAT1, --strat1 STRAT1
                        Name of the first strategy
  -r1 RUNS1 [RUNS1 ...], --runs1 RUNS1 [RUNS1 ...]
                        Paths to the first strategy runs results
  -s2 STRAT2, --strat2 STRAT2
                        Name of the second strategy
  -r2 RUNS2 [RUNS2 ...], --runs2 RUNS2 [RUNS2 ...]
                        Paths to the second strategy runs results
  -cp CONFIGS_PATH, --configs-path CONFIGS_PATH
                        Path to the compare configurations
  --savedir SAVEDIR     Path for saving the comparison results
```
To compare two results run

```bash
python3 compstrat.py \
    -s1 ALHPA -r1 mock_runs/strat_alpha.csv \
    -s2 BETA -r2 mock_runs/strat_beta.csv \
    -cp resources/compare_confs.yaml
```

Output example
![resources/alpha_beta_comp_vis_example.png](resources/alpha_beta_comp_vis_example.png)


## `generate_episodes` tool


This tool expands the dataset by creating new  episodes for each method

Docs:
```bash
usage: generate_episodes.py [-h] [-d DATASET]

options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Path to the dataset JSON file 
```
Run:
```bash
Use custom dataset path:
python3 generate_episodes.py -d <path-to-dataset>

Use default dataset path (../maps/DotNet/Maps/dataset.json):
python3 generate_episodes.py
```