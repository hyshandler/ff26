import numpy as np
import pandas as pd

from ff_model.quantile_model import QUANTILES
from ff_model.rf_model import predict_rf_quantiles, train_rf_quantile_model


def test_predicted_quantiles_are_monotonic_per_row() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 1, size=200)})
    y = pd.Series(X["x"] * 10 + rng.normal(0, 2, size=200))

    model = train_rf_quantile_model(X, y)
    predictions = predict_rf_quantiles(model, X)

    assert list(predictions.columns) == [f"p{int(q * 100)}" for q in QUANTILES]
    assert (predictions["p10"] <= predictions["p50"]).all()
    assert (predictions["p50"] <= predictions["p90"]).all()


def test_predictions_are_deterministic_given_fixed_random_state() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.uniform(0, 1, size=100)})
    y = pd.Series(X["x"] * 5 + rng.normal(0, 1, size=100))

    model_a = train_rf_quantile_model(X, y, random_state=0)
    model_b = train_rf_quantile_model(X, y, random_state=0)

    pd.testing.assert_frame_equal(predict_rf_quantiles(model_a, X), predict_rf_quantiles(model_b, X))
