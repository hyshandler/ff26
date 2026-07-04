from typing import Callable

import numpy as np
import pandas as pd

MetricFn = Callable[[pd.Series, pd.Series], float]


def mean_absolute_error(predicted: pd.Series, actual: pd.Series) -> float:
    """Average absolute difference between `predicted` and `actual`, aligned by index."""
    return (predicted - actual).abs().mean()


def spearman_rank_correlation(predicted: pd.Series, actual: pd.Series) -> float:
    """Spearman's rank correlation: how well `predicted`'s order matches `actual`'s.

    Used to compare rank-only signals (e.g. ADP) against `actual` where a
    points-based error metric like `mean_absolute_error` doesn't apply. Computed as
    the Pearson correlation of each series' ranks, which is equivalent to Spearman's
    rho without adding a scipy dependency.
    """
    return predicted.rank().corr(actual.rank())


def bootstrap_confidence_interval(
    predicted: pd.Series,
    actual: pd.Series,
    metric_fn: MetricFn,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for `metric_fn(predicted, actual)`.

    Resamples rows (paired, with replacement) `n_resamples` times, recomputes the
    metric each time, and returns the central `confidence` percentile band of the
    resampled values -- a way to see whether a metric difference (e.g. model vs.
    naive baseline MAE) is larger than the noise from a small backtest sample, not
    just report a single point estimate.
    """
    predicted = predicted.reset_index(drop=True)
    actual = actual.reset_index(drop=True)
    n = len(predicted)
    rng = np.random.default_rng(random_state)

    resampled_values = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resampled_values.append(
            metric_fn(predicted.iloc[indices].reset_index(drop=True), actual.iloc[indices].reset_index(drop=True))
        )

    alpha = (1 - confidence) / 2
    low, high = np.quantile(resampled_values, [alpha, 1 - alpha])
    return (float(low), float(high))


def per_split_metrics(
    df: pd.DataFrame, split_column: str, predicted_column: str, actual_column: str, metric_fn: MetricFn
) -> pd.Series:
    """`metric_fn` computed separately within each `split_column` group -- e.g. per
    walk-forward split, to see how much the metric varies season to season rather
    than only looking at the aggregate."""
    return df.groupby(split_column).apply(
        lambda g: metric_fn(g[predicted_column], g[actual_column])
    )


def leave_one_split_out_metrics(
    df: pd.DataFrame, split_column: str, predicted_column: str, actual_column: str, metric_fn: MetricFn
) -> pd.Series:
    """`metric_fn` computed on every row EXCEPT each `split_column` value in turn --
    how much the aggregate metric shifts when a single split (e.g. one season) is
    removed, as a check for whether one unusual split is driving the result."""
    results = {}
    for split_value in df[split_column].unique():
        remaining = df.loc[df[split_column] != split_value]
        results[split_value] = metric_fn(remaining[predicted_column], remaining[actual_column])
    return pd.Series(results)
