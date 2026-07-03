import pandas as pd
import pytest

from ff_model.evaluation import mean_absolute_error, spearman_rank_correlation


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
