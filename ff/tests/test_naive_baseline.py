import pandas as pd

from ff_model.naive_baseline import RAW_STAT_COLUMNS, predict_naive_baseline, score_naive_baseline
from ff_model.scoring import PPR


def test_predicts_the_trailing_average_of_raw_stats_through_the_training_season() -> None:
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2021, "week": 1, "carries": 10, "rushing_yards": 40,
             "rushing_tds": 0, "targets": 2, "receptions": 1, "receiving_yards": 5, "receiving_tds": 0},
            {"player_id": "p1", "season": 2022, "week": 1, "carries": 20, "rushing_yards": 100,
             "rushing_tds": 1, "targets": 4, "receptions": 3, "receiving_yards": 25, "receiving_tds": 1},
            {"player_id": "p1", "season": 2023, "week": 1, "carries": 999, "rushing_yards": 999,
             "rushing_tds": 9, "targets": 9, "receptions": 9, "receiving_yards": 999, "receiving_tds": 9},
        ]
    )

    result = predict_naive_baseline(weekly, player_ids={"p1"}, through_season=2022)

    row = result.set_index("player_id").loc["p1"]
    assert row["games"] == 2
    assert row["carries"] == 15.0
    assert row["rushing_yards"] == 70.0


def test_excludes_seasons_after_the_training_cutoff() -> None:
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2022, "week": 1, "carries": 10, "rushing_yards": 40,
             "rushing_tds": 0, "targets": 2, "receptions": 1, "receiving_yards": 5, "receiving_tds": 0},
            {"player_id": "p1", "season": 2023, "week": 1, "carries": 999, "rushing_yards": 999,
             "rushing_tds": 9, "targets": 9, "receptions": 9, "receiving_yards": 999, "receiving_tds": 9},
        ]
    )

    result = predict_naive_baseline(weekly, player_ids={"p1"}, through_season=2022)

    row = result.set_index("player_id").loc["p1"]
    assert row["carries"] == 10.0


def test_only_predicts_for_the_given_player_ids() -> None:
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2022, "week": 1, "carries": 10, "rushing_yards": 40,
             "rushing_tds": 0, "targets": 2, "receptions": 1, "receiving_yards": 5, "receiving_tds": 0},
            {"player_id": "p2", "season": 2022, "week": 1, "carries": 5, "rushing_yards": 20,
             "rushing_tds": 0, "targets": 1, "receptions": 1, "receiving_yards": 5, "receiving_tds": 0},
        ]
    )

    result = predict_naive_baseline(weekly, player_ids={"p1"}, through_season=2022)

    assert set(result["player_id"]) == {"p1"}


def test_predicts_trailing_average_for_a_custom_raw_stat_columns_list() -> None:
    weekly = pd.DataFrame(
        [
            {"player_id": "qb1", "season": 2021, "week": 1, "passing_yards": 200, "passing_tds": 1},
            {"player_id": "qb1", "season": 2022, "week": 1, "passing_yards": 300, "passing_tds": 3},
        ]
    )

    result = predict_naive_baseline(
        weekly,
        player_ids={"qb1"},
        through_season=2022,
        raw_stat_columns=["passing_yards", "passing_tds"],
    )

    row = result.set_index("player_id").loc["qb1"]
    assert row["passing_yards"] == 250.0
    assert row["passing_tds"] == 2.0
    assert "carries" not in result.columns


def test_score_naive_baseline_applies_the_scoring_formula_to_the_trailing_averages() -> None:
    predictions = pd.DataFrame(
        [{"player_id": "p1", "rushing_yards": 70.0, "rushing_tds": 0.5, "receptions": 2.0}]
    )

    points = score_naive_baseline(predictions, PPR)

    # PPR: 70 * 0.1 rushing_yards + 0.5 * 6 rushing_tds + 2 * 1 reception = 7 + 3 + 2 = 12.0
    assert points.loc[0] == 12.0


def test_output_has_the_documented_raw_stat_columns() -> None:
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2022, "week": 1, "carries": 10, "rushing_yards": 40,
             "rushing_tds": 0, "targets": 2, "receptions": 1, "receiving_yards": 5, "receiving_tds": 0},
        ]
    )

    result = predict_naive_baseline(weekly, player_ids={"p1"}, through_season=2022)

    assert set(RAW_STAT_COLUMNS) <= set(result.columns)
    assert {"player_id", "games"} <= set(result.columns)
