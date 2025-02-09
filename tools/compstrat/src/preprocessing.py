import pandas as pd


STRAT1_PREFIX = "strat1_"
STRAT2_PREFIX = "strat2_"


def preprocess(
    strat1_runs: list[pd.DataFrame], strat2_runs: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_runs = pd.concat(
        strat1_runs + strat2_runs,
        axis=1,
        join="inner",
        keys=list(map(lambda num: f"{STRAT1_PREFIX}{num}", range(len(strat1_runs))))
        + list(map(lambda num: f"{STRAT2_PREFIX}{num}", range(len(strat2_runs)))),
    )

    strat1_df, strat2_df = pd.DataFrame(), pd.DataFrame()
    metrics = strat1_runs[0].keys()

    for metric in metrics:
        strat1_df[metric] = all_runs.loc[
            :, [(f"{STRAT1_PREFIX}{num}", metric) for num in range(len(strat1_runs))]
        ].mean(axis=1)
        strat2_df[metric] = all_runs.loc[
            :, [(f"{STRAT2_PREFIX}{num}", metric) for num in range(len(strat2_runs))]
        ].mean(axis=1)

        strat1_df[f"{metric}_min"] = pd.concat(
            [all_runs[f"{STRAT1_PREFIX}{i}"][metric] for i in range(5)],
            axis=1,
            keys=list(range(5)),
        ).min(axis=1)
        strat2_df[f"{metric}_min"] = pd.concat(
            [all_runs[f"{STRAT2_PREFIX}{i}"][metric] for i in range(5)],
            axis=1,
            keys=list(range(5)),
        ).min(axis=1)
        strat1_df[f"{metric}_max"] = pd.concat(
            [all_runs[f"{STRAT1_PREFIX}{i}"][metric] for i in range(5)],
            axis=1,
            keys=list(range(5)),
        ).max(axis=1)
        strat2_df[f"{metric}_max"] = pd.concat(
            [all_runs[f"{STRAT2_PREFIX}{i}"][metric] for i in range(5)],
            axis=1,
            keys=list(range(5)),
        ).max(axis=1)
    return strat1_df, strat2_df
