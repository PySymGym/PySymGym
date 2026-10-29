import pandas as pd


STRAT1_PREFIX = "strat1_"
STRAT2_PREFIX = "strat2_"
MIN_POSTFIX, MAX_POSTFIX = "_min", "_max"


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

        def get_mean_by_key(keys, metric):
            return all_runs.loc[:, [(key, metric) for key in keys]].mean(axis=1)

        strat1_df[metric], strat2_df[metric] = (
            get_mean_by_key(strat1_keys, metric),
            get_mean_by_key(strat2_keys, metric),
        )

        strat1_single_metric, strat2_single_metric = (
            pd.concat(
                [all_runs[strat1_key][metric] for strat1_key in strat1_keys],
                axis=1,
                keys=list(range(len(strat1_runs))),
            ),
            pd.concat(
                [all_runs[strat2_key][metric] for strat2_key in strat2_keys],
                axis=1,
                keys=list(range(len(strat2_runs))),
            ),
        )

        strat1_df[f"{metric}{MIN_POSTFIX}"], strat1_df[f"{metric}{MAX_POSTFIX}"] = (
            strat1_single_metric.min(axis=1),
            strat1_single_metric.max(axis=1),
        )
        strat2_df[f"{metric}{MIN_POSTFIX}"], strat2_df[f"{metric}{MAX_POSTFIX}"] = (
            strat2_single_metric.min(axis=1),
            strat2_single_metric.max(axis=1),
        )
    return strat1_df, strat2_df
