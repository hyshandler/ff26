import pandas as pd
import pytest

from ff_model.evaluation import (
    bootstrap_confidence_interval,
    leave_one_split_out_metrics,
    mean_absolute_error,
    per_split_metrics,
    spearman_rank_correlation,
)


def test_mean_absolute_error_averages_the_absolute_differences() -> None:
    predicted = pd.Series([10.0, 20.0, 0.0])
    actual = pd.Series([12.0, 15.0, 5.0])

    # |10-12| + |20-15| + |0-5| = 2 + 5 + 5 = 12, / 3 = 4.0
    assert mean_absolute_error(predicted, actual) == 4.0


def test_spearman_rank_correlation_is_one_for_a_perfectly_matching_order() -> None:
    predicted = pd.Series([1.0, 2.0, 3.0, 4.0])
    actual = pd.Series([10.0, 20.0, 30.0, 40.0])

    assert spearman_rank_correlation(predicted, actual) == 1.0


def test_spearman_rank_correlation_is_negative_one_for_a_fully_reversed_order() -> None:
    predicted = pd.Series([1.0, 2.0, 3.0, 4.0])
    actual = pd.Series([40.0, 30.0, 20.0, 10.0])

    assert spearman_rank_correlation(predicted, actual) == -1.0


def test_spearman_rank_correlation_for_a_partially_matching_order() -> None:
    predicted = pd.Series([1.0, 2.0, 3.0, 4.0])
    actual = pd.Series([2.0, 1.0, 4.0, 3.0])

    # Worked by hand: rank pairs (1,2),(2,1),(3,4),(4,3) -> Pearson r of ranks = 0.6
    assert spearman_rank_correlation(predicted, actual) == pytest.approx(0.6)


def test_bootstrap_confidence_interval_is_a_single_point_for_a_perfect_predictor() -> None:
    predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    low, high = bootstrap_confidence_interval(
        predicted, actual, mean_absolute_error, n_resamples=200, random_state=0
    )

    # Every resample duplicates rows from a perfectly-matching series, so MAE is
    # exactly 0 in every resample -- the interval collapses to a point.
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_bootstrap_confidence_interval_is_deterministic_given_a_random_state() -> None:
    predicted = pd.Series([10.0, 8.0, 15.0, 2.0, 30.0, 4.0, 21.0])
    actual = pd.Series([12.0, 5.0, 14.0, 6.0, 25.0, 9.0, 18.0])

    first = bootstrap_confidence_interval(
        predicted, actual, mean_absolute_error, n_resamples=200, random_state=42
    )
    second = bootstrap_confidence_interval(
        predicted, actual, mean_absolute_error, n_resamples=200, random_state=42
    )

    assert first == second


def test_bootstrap_confidence_interval_brackets_the_point_estimate() -> None:
    predicted = pd.Series([10.0, 8.0, 15.0, 2.0, 30.0, 4.0, 21.0])
    actual = pd.Series([12.0, 5.0, 14.0, 6.0, 25.0, 9.0, 18.0])

    point_estimate = mean_absolute_error(predicted, actual)
    low, high = bootstrap_confidence_interval(
        predicted, actual, mean_absolute_error, n_resamples=500, random_state=0
    )

    assert low <= point_estimate <= high


def test_per_split_metrics_computes_the_metric_within_each_split() -> None:
    df = pd.DataFrame(
        {
            "target_season": [2022, 2022, 2023, 2023],
            "predicted": [10.0, 20.0, 0.0, 0.0],
            "actual": [12.0, 15.0, 5.0, 5.0],
        }
    )

    result = per_split_metrics(df, "target_season", "predicted", "actual", mean_absolute_error)

    # 2022: |10-12| + |20-15| = 2 + 5, /2 = 3.5. 2023: |0-5| + |0-5| = 10, /2 = 5.0
    assert result[2022] == pytest.approx(3.5)
    assert result[2023] == pytest.approx(5.0)


def test_leave_one_split_out_metrics_excludes_each_split_in_turn() -> None:
    df = pd.DataFrame(
        {
            "target_season": [2021, 2021, 2022, 2022],
            "predicted": [10.0, 10.0, 0.0, 0.0],
            "actual": [10.0, 10.0, 100.0, 100.0],
        }
    )

    result = leave_one_split_out_metrics(
        df, "target_season", "predicted", "actual", mean_absolute_error
    )

    # Excluding 2021 (the perfect rows) leaves only 2022's huge miss -> MAE 100.
    assert result[2021] == pytest.approx(100.0)
    # Excluding 2022 (the huge miss) leaves only 2021's perfect rows -> MAE 0.
    assert result[2022] == pytest.approx(0.0)
