import numpy as np
import pandas as pd
import pytest

from ff_model.quantile_model import QUANTILES
from ff_model.tabfm_model import predict_tabfm_quantiles, train_tabfm_model


@pytest.mark.network
def test_predicted_quantiles_are_monotonic_per_row() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 1, size=50)})
    y = pd.Series(X["x"] * 10 + rng.normal(0, 2, size=50))

    model = train_tabfm_model(X, y)
    predictions = predict_tabfm_quantiles(model, X)

    assert list(predictions.columns) == [f"p{int(q * 100)}" for q in QUANTILES]
    assert (predictions["p10"] <= predictions["p50"]).all()
    assert (predictions["p50"] <= predictions["p90"]).all()
