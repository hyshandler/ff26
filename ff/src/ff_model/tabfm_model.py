import pandas as pd
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from ff_model.quantile_model import QUANTILES


def train_tabfm_model(X: pd.DataFrame, y: pd.Series) -> TabPFNRegressor:
    """Fit one TabFM model; unlike LightGBM it predicts all quantiles from a single fit.

    Pinned to ModelVersion.V2, the original open-weights release: it downloads
    unauthenticated from HuggingFace, unlike the default (v2.6/v3) which requires
    an interactive browser-based license acceptance this pipeline can't perform.
    """
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu")
    model.fit(X, y)
    return model


def predict_tabfm_quantiles(model: TabPFNRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Predict P10/P50/P90, sorted per row so quantiles are always monotonic."""
    raw = model.predict(X, output_type="quantiles", quantiles=list(QUANTILES))
    stacked = pd.DataFrame(raw, index=[f"p{int(q * 100)}" for q in QUANTILES]).T
    sorted_values = stacked.to_numpy().copy()
    sorted_values.sort(axis=1)
    columns = [f"p{int(q * 100)}" for q in QUANTILES]
    return pd.DataFrame(sorted_values, columns=columns, index=X.index)
