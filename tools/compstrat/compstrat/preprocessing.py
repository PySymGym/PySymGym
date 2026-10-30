import warnings

import pandas as pd


STRAT1_PREFIX = "strat1_"
STRAT2_PREFIX = "strat2_"
MIN_POSTFIX, MAX_POSTFIX = "_min", "_max"
_MAX_LISTED_METHODS = 10


def _union_index(runs: list[pd.DataFrame]) -> pd.Index:
    if not runs:
        return pd.Index([])
    union = runs[0].index
    for run in runs[1:]:
        union = union.union(run.index)
    return union


def _format_index(index: pd.Index) -> str:
    # Sorted: Index.difference orders its result by hash, which is
    # nondeterministic across runs.
    names = sorted(str(name) for name in index)
    shown = ", ".join(names[:_MAX_LISTED_METHODS])
    if len(names) > _MAX_LISTED_METHODS:
        shown += f", ... and {len(names) - _MAX_LISTED_METHODS} more"
    return shown


def _single_metric(
    all_runs: pd.DataFrame, keys: list[str], metric: str
) -> pd.DataFrame:
    """The per-run values of one metric, skipping runs that lack it."""
    parts = [all_runs[key][metric] for key in keys if metric in all_runs[key].columns]
    if not parts:
        return pd.DataFrame(index=all_runs.index)
    return pd.concat(parts, axis=1, keys=list(range(len(parts))))


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

    # The inner join keeps only the methods present in every run of both
    # strategies; make the silent truncation visible.
    strat1_only = _union_index(strat1_runs).difference(_union_index(strat2_runs))
    strat2_only = _union_index(strat2_runs).difference(_union_index(strat1_runs))
    if len(strat1_only) or len(strat2_only):
        warnings.warn(
            "inner join drops methods present in only one strategy's runs: "
            f"only in {STRAT1_PREFIX.rstrip('_')}: {_format_index(strat1_only)}; "
            f"only in {STRAT2_PREFIX.rstrip('_')}: {_format_index(strat2_only)}",
            stacklevel=2,
        )

    strat1_df, strat2_df = pd.DataFrame(), pd.DataFrame()
    # The compared metrics are the union of both strategies' run columns
    # (first-seen order): a column present in only one strategy's runs is
    # still compared, with NaN on the missing side.
    metrics = list(
        dict.fromkeys(
            column for run in strat1_runs + strat2_runs for column in run.columns
        )
    )

    for metric in metrics:

        def get_mean_by_key(keys, metric):
            cols = [(key, metric) for key in keys if (key, metric) in all_runs.columns]
            if not cols:
                return pd.Series(index=all_runs.index, dtype="float64")
            return all_runs.loc[:, cols].mean(axis=1)

        strat1_df[metric], strat2_df[metric] = (
            get_mean_by_key(strat1_keys, metric),
            get_mean_by_key(strat2_keys, metric),
        )

        strat1_single_metric, strat2_single_metric = (
            _single_metric(all_runs, strat1_keys, metric),
            _single_metric(all_runs, strat2_keys, metric),
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
