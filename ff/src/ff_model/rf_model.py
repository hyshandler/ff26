import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ff_model.quantile_model import QUANTILES


def train_rf_quantile_model(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> RandomForestRegressor:
    """Fit one RandomForest whose per-tree predictions double as a quantile ensemble.

    Unlike LightGBM's native quantile objective (one model per quantile), a single
    forest fit yields every tree's leaf prediction per row; quantiles across that
    per-row distribution (see `predict_rf_quantiles`) approximate a quantile
    regression forest without needing separate fits per quantile.
    """
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X, y)
    return model


def predict_rf_quantiles(model: RandomForestRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Predict P10/P50/P90 from the spread of individual trees' predictions per row."""
    per_tree = np.column_stack([tree.predict(X.to_numpy()) for tree in model.estimators_])
    raw = np.quantile(per_tree, QUANTILES, axis=1).T
    columns = [f"p{int(q * 100)}" for q in QUANTILES]
    return pd.DataFrame(raw, columns=columns, index=X.index)
