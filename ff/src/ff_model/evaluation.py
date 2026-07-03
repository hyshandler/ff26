import pandas as pd


def mean_absolute_error(predicted: pd.Series, actual: pd.Series) -> float:
    """Average absolute difference between `predicted` and `actual`, aligned by index."""
    return (predicted - actual).abs().mean()
