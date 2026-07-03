import pandas as pd


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
