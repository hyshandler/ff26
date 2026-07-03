import pandas as pd
import pytest

from ff_model.run_backtest import run_backtest, with_actual_outcomes


def test_joins_actual_games_played_and_actual_fantasy_points_for_the_target_season() -> None:
    backtest_result = pd.DataFrame(
        [{"player_id": "p1", "target_season": 2023, "full_projection_p50": 100.0}]
    )
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2023, "week": 1, "rushing_yards": 100, "rushing_tds": 1},
            {"player_id": "p1", "season": 2023, "week": 2, "rushing_yards": 50, "rushing_tds": 0},
        ]
    )

    result = with_actual_outcomes(backtest_result, weekly)

    row = result.loc[0]
    assert row["actual_games_played"] == 2
    # PPR: (100 + 50) * 0.1 rushing_yards + 1 * 6 rushing_tds = 21.0
    assert row["actual_fantasy_points"] == 21.0


def test_a_player_absent_from_weekly_in_the_target_season_gets_zero_actual_outcomes() -> None:
    """Per ADR-0008: a player who got hurt and never returned must still appear with a
    real (zero) outcome, not be dropped -- dropping them would survivorship-bias the
    backtest comparison toward players who stayed healthy."""
    backtest_result = pd.DataFrame(
        [{"player_id": "p1", "target_season": 2023, "full_projection_p50": 100.0}]
    )
    weekly = pd.DataFrame(columns=["player_id", "season", "week", "rushing_yards", "rushing_tds"])

    result = with_actual_outcomes(backtest_result, weekly)

    row = result.loc[0]
    assert row["actual_games_played"] == 0
    assert row["actual_fantasy_points"] == 0.0


@pytest.mark.network
def test_concatenates_every_split_into_one_dataframe_with_a_split_marker() -> None:
    result = run_backtest("RB", seasons=[2020, 2021, 2022, 2023], min_train_seasons=2)

    assert isinstance(result, pd.DataFrame)
    assert set(result["target_season"]) == {2022, 2023}

    # Each split's rows are traceable back to the season boundary that produced them.
    by_split = result.groupby("target_season")["train_through_season"].unique()
    assert by_split[2022].tolist() == [2021]
    assert by_split[2023].tolist() == [2022]
