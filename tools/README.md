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
usage: compstrat.py [-h] -s1 STRAT1 -r1 RUN1_0 RUN1_1 RUN1_2 ... -s2 STRAT2 -r2 RUN2_0 RUN2_1 RUN2_2 ... -cp CONFIGS_PATH [--savedir SAVEDIR]

options:
  -h, --help            show this help message and exit
  -s1 STRAT1, --strat1 STRAT1
                        Name of the first strategy
  -r1 RUN1_0 RUN1_1 RUN1_2 ..., --runs1 RUN1_0 RUN1_1 RUN1_2 ...
                        Path to the first strategy run result
  -s2 STRAT2, --strat2 STRAT2
                        Name of the second strategy
  -r2 RUN2_0 RUN2_1 RUN2_2 ..., --runs2 RUN2_0 RUN2_1 RUN2_2 ...
                        Path to ther second strategy run result
  -cp CONFIGS_PATH, --configs-path CONFIGS_PATH
                        Path to ther second strategy run result
  --savedir SAVEDIR     Path to save results to
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
