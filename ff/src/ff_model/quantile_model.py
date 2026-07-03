import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def train_quantile_models(X: pd.DataFrame, y: pd.Series) -> dict[float, LGBMRegressor]:
    """Train one LightGBM model per quantile, using LightGBM's native quantile objective."""
    models = {}
    for quantile in QUANTILES:
        model = LGBMRegressor(objective="quantile", alpha=quantile, verbosity=-1)
        model.fit(X, y)
        models[quantile] = model
    return models


def predict_quantiles(models: dict[float, LGBMRegressor], X: pd.DataFrame) -> pd.DataFrame:
    """Predict P10/P50/P90, sorted per row so quantiles are always monotonic.

    Independently-trained quantile regressors aren't guaranteed to agree on
    ordering for every row; sorting is the standard fix (quantile crossing).
    """
    raw = np.column_stack([models[q].predict(X) for q in QUANTILES])
    raw.sort(axis=1)
    columns = [f"p{int(q * 100)}" for q in QUANTILES]
    return pd.DataFrame(raw, columns=columns, index=X.index)
