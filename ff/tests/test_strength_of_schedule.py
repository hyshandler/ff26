import pandas as pd
import pytest

from ff_model.scoring import PPR
from ff_model.strength_of_schedule import (
    add_actual_game_sos,
    season_ending_points_allowed,
    season_wide_sos,
    season_wide_sos_by_player,
    season_wide_sos_history,
    trailing_points_allowed,
    weekly_points_allowed,
)


def _weekly(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_weekly_points_allowed_sums_fantasy_points_scored_against_each_defense() -> None:
    weekly = _weekly(
        [
            {
                "season": 2023,
                "week": 1,
                "opponent_team": "NYJ",
                "position": "RB",
                "rushing_yards": 100,
                "rushing_tds": 1,
            },
            {
                "season": 2023,
                "week": 1,
                "opponent_team": "NYJ",
                "position": "RB",
                "rushing_yards": 20,
                "rushing_tds": 0,
            },
        ]
    )

    result = weekly_points_allowed(weekly, PPR)

    row = result.set_index(["season", "week", "team", "position"]).loc[(2023, 1, "NYJ", "RB")]
    # (100+20)*0.1 rushing_yards + 1*6 rushing_td = 18.0
    assert row["points_allowed"] == pytest.approx(18.0)


def test_trailing_points_allowed_is_blind_to_the_current_and_later_weeks() -> None:
    points_allowed = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "team": "NYJ", "position": "RB", "points_allowed": 10.0},
            {"season": 2023, "week": 2, "team": "NYJ", "position": "RB", "points_allowed": 30.0},
            {"season": 2023, "week": 3, "team": "NYJ", "position": "RB", "points_allowed": 999.0},
        ]
    )

    result = trailing_points_allowed(points_allowed)

    week1 = result.loc[result["week"] == 1, "trailing_points_allowed"]
    assert week1.isna().all()

    week2 = result.loc[result["week"] == 2, "trailing_points_allowed"].item()
    assert week2 == 10.0

    week3 = result.loc[result["week"] == 3, "trailing_points_allowed"].item()
    assert week3 == 20.0


def test_add_actual_game_sos_joins_each_players_own_opponent_that_week() -> None:
    weekly = _weekly(
        [
            {
                "season": 2023,
                "week": 2,
                "player_id": "p1",
                "position": "RB",
                "opponent_team": "NYJ",
            }
        ]
    )
    trailing = pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 2,
                "team": "NYJ",
                "position": "RB",
                "trailing_points_allowed": 15.0,
            },
            {
                "season": 2023,
                "week": 2,
                "team": "MIA",
                "position": "RB",
                "trailing_points_allowed": 999.0,
            },
        ]
    )

    result = add_actual_game_sos(weekly, trailing)

    assert result.loc[0, "trailing_points_allowed"] == 15.0


def test_season_ending_points_allowed_averages_a_completed_seasons_weekly_values() -> None:
    points_allowed = pd.DataFrame(
        [
            {"season": 2022, "week": 1, "team": "NYJ", "position": "RB", "points_allowed": 10.0},
            {"season": 2022, "week": 2, "team": "NYJ", "position": "RB", "points_allowed": 20.0},
        ]
    )

    result = season_ending_points_allowed(points_allowed, season=2022)

    row = result.set_index(["team", "position"]).loc[("NYJ", "RB")]
    assert row["points_allowed_per_game"] == 15.0


def test_season_wide_sos_averages_opponent_strength_across_the_full_schedule() -> None:
    schedule = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "team": "BUF", "opponent_team": "NYJ"},
            {"season": 2023, "week": 2, "team": "BUF", "opponent_team": "MIA"},
        ]
    )
    season_ending = pd.DataFrame(
        [
            {"team": "NYJ", "position": "RB", "points_allowed_per_game": 10.0},
            {"team": "MIA", "position": "RB", "points_allowed_per_game": 20.0},
        ]
    )

    result = season_wide_sos(schedule, season_ending, position="RB", season=2023)

    row = result.set_index("team").loc["BUF"]
    assert row["season_wide_sos"] == 15.0


def test_season_wide_sos_by_player_joins_via_the_players_roster_team_and_season() -> None:
    sos_by_team = pd.DataFrame(
        [
            {"season": 2022, "team": "BUF", "season_wide_sos": 999.0},
            {"season": 2023, "team": "BUF", "season_wide_sos": 15.0},
        ]
    )
    rosters = pd.DataFrame([{"player_id": "p1", "season": 2023, "team": "BUF"}])

    result = season_wide_sos_by_player(sos_by_team, rosters)

    row = result.set_index(["season", "player_id"]).loc[(2023, "p1")]
    assert row["season_wide_sos"] == 15.0


def test_season_wide_sos_by_player_deduplicates_a_player_traded_mid_season() -> None:
    sos_by_team = pd.DataFrame(
        [
            {"season": 2023, "team": "BUF", "season_wide_sos": 15.0},
            {"season": 2023, "team": "MIA", "season_wide_sos": 25.0},
        ]
    )
    rosters = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2023, "team": "BUF"},
            {"player_id": "p1", "season": 2023, "team": "MIA"},
        ]
    )

    result = season_wide_sos_by_player(sos_by_team, rosters)

    assert len(result) == 1


def test_season_wide_sos_history_uses_the_prior_seasons_completed_defense_data() -> None:
    schedule = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "team": "BUF", "opponent_team": "NYJ"},
        ]
    )
    points_allowed = pd.DataFrame(
        [
            {"season": 2022, "week": 1, "team": "NYJ", "position": "RB", "points_allowed": 10.0},
            {"season": 2022, "week": 2, "team": "NYJ", "position": "RB", "points_allowed": 20.0},
            # 2023's own (not-yet-known-in-advance) data must NOT be used.
            {"season": 2023, "week": 1, "team": "NYJ", "position": "RB", "points_allowed": 999.0},
        ]
    )

    result = season_wide_sos_history(schedule, points_allowed, position="RB", seasons=[2022, 2023])

    row = result.set_index("team").loc["BUF"]
    assert row["season_wide_sos"] == 15.0
    # 2022 has no prior completed season to draw opponent strength from.
    assert 2022 not in set(result["season"])
