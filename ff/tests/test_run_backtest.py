import pandas as pd
import pytest

from ff_model.adp import crosswalk_adp_to_player_ids, load_adp
from ff_model.nflverse import load_seasonal_rosters
from ff_model.run_backtest import run_backtest, with_actual_outcomes, with_adp_benchmark


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


def test_joins_adp_onto_backtest_rows_matching_the_adps_season() -> None:
    backtest_result = pd.DataFrame(
        [
            {"player_id": "p1", "target_season": 2022, "full_projection_p50": 100.0},
            {"player_id": "p1", "target_season": 2023, "full_projection_p50": 110.0},
        ]
    )
    adp = pd.DataFrame([{"name": "Player One", "position": "RB", "player_id": "p1", "adp": 14.9}])

    result = with_adp_benchmark(backtest_result, adp, season=2022)

    by_season = result.set_index("target_season")
    assert by_season.loc[2022, "adp"] == 14.9
    assert pd.isna(by_season.loc[2023, "adp"])


def test_a_player_missing_from_the_adp_crosswalk_gets_a_nan_adp() -> None:
    backtest_result = pd.DataFrame(
        [{"player_id": "p1", "target_season": 2022, "full_projection_p50": 100.0}]
    )
    adp = pd.DataFrame([{"name": "Someone Else", "position": "RB", "player_id": pd.NA, "adp": 50.0}])

    result = with_adp_benchmark(backtest_result, adp, season=2022)

    assert pd.isna(result.loc[0, "adp"])


@pytest.mark.network
def test_concatenates_every_split_into_one_dataframe_with_a_split_marker() -> None:
    result = run_backtest("RB", seasons=[2020, 2021, 2022, 2023], min_train_seasons=2)

    assert isinstance(result, pd.DataFrame)
    assert set(result["target_season"]) == {2022, 2023}

    # Each split's rows are traceable back to the season boundary that produced them.
    by_split = result.groupby("target_season")["train_through_season"].unique()
    assert by_split[2022].tolist() == [2021]
    assert by_split[2023].tolist() == [2022]


@pytest.mark.network
def test_with_adp_benchmark_joins_real_ffc_adp_onto_a_real_backtest_result() -> None:
    result = run_backtest("RB", seasons=[2020, 2021, 2022, 2023], min_train_seasons=2)

    rosters = load_seasonal_rosters([2022])
    adp = crosswalk_adp_to_player_ids(load_adp(2022), rosters, season=2022)

    with_adp = with_adp_benchmark(result, adp, season=2022)

    assert "adp" in with_adp.columns
    season_2022 = with_adp.loc[with_adp["target_season"] == 2022]
    assert season_2022["adp"].notna().any()
    assert (season_2022["adp"].dropna() > 0).all()
    # The other split's rows are untouched by this season's ADP.
    assert with_adp.loc[with_adp["target_season"] == 2023, "adp"].isna().all()
