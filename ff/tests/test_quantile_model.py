import numpy as np
import pandas as pd

from ff_model.quantile_model import QUANTILES, predict_quantiles, train_quantile_models


def test_predicted_quantiles_are_monotonic_per_row() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 1, size=200)})
    y = pd.Series(X["x"] * 10 + rng.normal(0, 2, size=200))

    models = train_quantile_models(X, y)
    predictions = predict_quantiles(models, X)

    assert list(predictions.columns) == [f"p{int(q * 100)}" for q in QUANTILES]
    assert (predictions["p10"] <= predictions["p50"]).all()
    assert (predictions["p50"] <= predictions["p90"]).all()


def test_trains_one_model_per_quantile() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    models = train_quantile_models(X, y)

    assert set(models.keys()) == set(QUANTILES)
