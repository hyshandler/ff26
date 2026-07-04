import pandas as pd
import pytest

from ff_model.feature_audit import (
    add_noise_control_column,
    feature_correlation_matrix,
    features_worse_than_noise,
    highly_correlated_pairs,
)


def test_feature_correlation_matrix_is_symmetric_with_ones_on_the_diagonal() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = feature_correlation_matrix(df, ["a", "b", "c"])

    assert result.loc["a", "a"] == pytest.approx(1.0)
    assert result.loc["a", "b"] == pytest.approx(-1.0)
    # A constant column has undefined correlation, not zero -- left as NaN, not imputed.
    assert pd.isna(result.loc["a", "c"])


def test_highly_correlated_pairs_flags_pairs_above_the_threshold_once_each() -> None:
    corr = pd.DataFrame(
        {
            "a": [1.0, 0.95, 0.1],
            "b": [0.95, 1.0, 0.2],
            "c": [0.1, 0.2, 1.0],
        },
        index=["a", "b", "c"],
    )

    result = highly_correlated_pairs(corr, threshold=0.8)

    assert result == [("a", "b", pytest.approx(0.95))]


def test_highly_correlated_pairs_respects_a_lower_threshold() -> None:
    corr = pd.DataFrame(
        {
            "a": [1.0, 0.5, 0.1],
            "b": [0.5, 1.0, 0.85],
            "c": [0.1, 0.85, 1.0],
        },
        index=["a", "b", "c"],
    )

    result = highly_correlated_pairs(corr, threshold=0.4)

    assert ("a", "b", pytest.approx(0.5)) in result
    assert ("b", "c", pytest.approx(0.85)) in result
    assert len(result) == 2


def test_add_noise_control_column_adds_an_int_column_not_derived_from_the_data() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    result = add_noise_control_column(df, random_state=0)

    assert "noise_control" in result.columns
    assert len(result) == 3
    assert pd.api.types.is_integer_dtype(result["noise_control"])
    # Original columns and row count are untouched.
    assert result["a"].tolist() == [1.0, 2.0, 3.0]


def test_add_noise_control_column_is_deterministic_given_a_random_state() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    first = add_noise_control_column(df, random_state=0)
    second = add_noise_control_column(df, random_state=0)

    assert first["noise_control"].tolist() == second["noise_control"].tolist()


def test_features_worse_than_noise_flags_only_features_at_or_below_the_noise_columns_importance() -> (
    None
):
    importances = pd.Series(
        {"good_feature": 50.0, "weak_feature": 2.0, "noise_control": 5.0, "great_feature": 100.0}
    )

    result = features_worse_than_noise(importances, noise_column="noise_control")

    assert result == ["weak_feature"]
