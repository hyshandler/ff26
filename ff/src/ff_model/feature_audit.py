import numpy as np
import pandas as pd


def feature_correlation_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Pairwise Pearson correlation among `feature_columns`, for spotting redundant
    (highly collinear) features before adding new ones to a position's feature set."""
    return df[feature_columns].corr()


def highly_correlated_pairs(
    corr: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """Every unordered pair of features whose absolute correlation exceeds `threshold`,
    each listed once, sorted by descending absolute correlation."""
    pairs = []
    columns = list(corr.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) > threshold:
                pairs.append((a, b, value))
    return sorted(pairs, key=lambda pair: -abs(pair[2]))


def add_noise_control_column(
    df: pd.DataFrame, column_name: str = "noise_control", random_state: int = 0
) -> pd.DataFrame:
    """Adds a column of random ints, independent of every other column, so it can be fed
    into training alongside real features: any real feature whose LightGBM importance
    doesn't clear this column's is providing no more signal than noise (see
    `features_worse_than_noise`)."""
    rng = np.random.default_rng(random_state)
    result = df.copy()
    result[column_name] = rng.integers(0, 1_000_000, size=len(df))
    return result


def features_worse_than_noise(
    importances: pd.Series, noise_column: str = "noise_control"
) -> list[str]:
    """Every feature (other than `noise_column` itself) whose importance is at or below
    the noise column's -- i.e. no better than a column of random ints."""
    noise_importance = importances[noise_column]
    return [
        feature
        for feature, importance in importances.items()
        if feature != noise_column and importance <= noise_importance
    ]
