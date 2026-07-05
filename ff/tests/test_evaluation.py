import pandas as pd
import pytest

from ff_model.evaluation import (
    bootstrap_confidence_interval,
    leave_one_split_out_metrics,
    matched_population_report,
    mean_absolute_error,
    per_split_metrics,
    season_block_bootstrap_interval,
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


def test_matched_population_report_computes_each_predictor_on_the_identical_matched_rows() -> None:
    """Per ADR-0010: model, ADP, and naive rho must all be computed on the same
    row set (the Matched Population), not each predictor's own non-null subset."""
    df = pd.DataFrame(
        [
            {"model": 4.0, "neg_adp": -1.0, "naive": 3.0, "actual": 40.0, "matched": True},
            {"model": 3.0, "neg_adp": -2.0, "naive": 4.0, "actual": 30.0, "matched": True},
            {"model": 2.0, "neg_adp": -3.0, "naive": 1.0, "actual": 20.0, "matched": True},
            {"model": 1.0, "neg_adp": -4.0, "naive": 2.0, "actual": 10.0, "matched": True},
            # Unmatched row: no ADP at all, huge model rank noise if ever mixed in.
            {"model": 999.0, "neg_adp": None, "naive": 999.0, "actual": 1.0, "matched": False},
        ]
    )

    report = matched_population_report(
        df,
        actual_column="actual",
        matched_mask=df["matched"],
        prediction_columns={"model": "model", "adp": "neg_adp", "naive": "naive"},
    )

    assert report["n_matched"] == 4
    # model/neg_adp/actual are all perfectly co-ranked on the 4 matched rows -> rho 1.0.
    assert report["matched_population"]["model"] == pytest.approx(1.0)
    assert report["matched_population"]["adp"] == pytest.approx(1.0)
    # naive's rank order on the matched rows (3,4,1,2) vs actual's (4,3,2,1) is imperfect.
    assert report["matched_population"]["naive"] == pytest.approx(0.6)
    # Full-population numbers are reported too, but under a clearly separate context-only key.
    assert set(report["full_population_context_only"]) == {"model", "adp", "naive"}
    assert report["full_population_context_only"]["model"] != report["matched_population"]["model"]


def test_season_block_bootstrap_interval_resamples_whole_seasons_not_rows() -> None:
    """Per ADR-0014: ~14 backtest seasons, not individual players, is the real unit of
    independence, so confidence intervals must resample whole season blocks."""
    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "value": [10.0, 10.0, 10.0, 100.0, 100.0, 100.0],
        }
    )

    low, high = season_block_bootstrap_interval(
        df, "season", metric_fn=lambda block: block["value"].mean(), n_resamples=1000, random_state=0
    )

    # Every possible resample is built entirely from whole 2020 (all 10s) and/or 2021
    # (all 100s) blocks, so the mean of any resample can only ever land on 10, 55, or 100
    # -- never some other value a row-level shuffle could produce.
    assert 10.0 <= low <= high <= 100.0


def test_season_block_bootstrap_interval_is_a_point_for_a_single_season() -> None:
    df = pd.DataFrame({"season": [2022, 2022, 2022], "value": [5.0, 7.0, 9.0]})

    low, high = season_block_bootstrap_interval(df, "season", metric_fn=lambda block: block["value"].mean())

    # Only one season exists, so every resample is that season repeated -- no variance.
    assert low == pytest.approx(7.0)
    assert high == pytest.approx(7.0)
