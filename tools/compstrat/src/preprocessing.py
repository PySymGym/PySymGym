import pandas as pd


STRAT1_PREFIX = "strat1_"
STRAT2_PREFIX = "strat2_"


def preprocess(
    strat1_runs: list[pd.DataFrame], strat2_runs: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat1_keys, strat2_keys = (
        [f"{STRAT1_PREFIX}{num}" for num in range(len(strat1_runs))],
        [f"{STRAT2_PREFIX}{num}" for num in range(len(strat2_runs))],
    )
    all_runs = pd.concat(
        strat1_runs + strat2_runs,
        axis=1,
        join="inner",
        keys=strat1_keys + strat2_keys,
    )

    strat1_df, strat2_df = pd.DataFrame(), pd.DataFrame()
    metrics = strat1_runs[0].keys()

    for metric in metrics:
        strat1_df[metric] = all_runs.loc[
            :, [(strat1_key, metric) for strat1_key in strat1_keys]
        ].mean(axis=1)
        strat2_df[metric] = all_runs.loc[
            :, [(strat2_key, metric) for strat2_key in strat2_keys]
        ].mean(axis=1)

        strat1_df[f"{metric}_min"] = pd.concat(
            [all_runs[strat1_key][metric] for strat1_key in strat1_keys],
            axis=1,
            keys=list(range(len(strat1_runs))),
        ).min(axis=1)
        strat2_df[f"{metric}_min"] = pd.concat(
            [all_runs[strat2_key][metric] for strat2_key in strat2_keys],
            axis=1,
            keys=list(range(len(strat2_runs))),
        ).min(axis=1)
        strat1_df[f"{metric}_max"] = pd.concat(
            [all_runs[strat1_key][metric] for strat1_key in strat1_keys],
            axis=1,
            keys=list(range(len(strat1_runs))),
        ).max(axis=1)
        strat2_df[f"{metric}_max"] = pd.concat(
            [all_runs[strat2_key][metric] for strat2_key in strat2_keys],
            axis=1,
            keys=list(range(len(strat2_runs))),
        ).max(axis=1)
    return strat1_df, strat2_df
