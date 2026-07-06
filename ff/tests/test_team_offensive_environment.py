import pandas as pd
import pytest

from ff_model.team_offensive_environment import (
    season_ending_team_offensive_environment,
    team_offensive_environment_by_player,
    team_offensive_environment_history,
    team_weekly_passing,
)


def test_team_weekly_passing_sums_attempts_and_yards_by_team_and_week() -> None:
    weekly = pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 1,
                "recent_team": "BUF",
                "attempts": 30,
                "passing_yards": 250,
            },
            {
                "season": 2023,
                "week": 1,
                "recent_team": "BUF",
                "attempts": 2,
                "passing_yards": 15,
            },
        ]
    )

    result = team_weekly_passing(weekly)

    row = result.set_index(["season", "week", "team"]).loc[(2023, 1, "BUF")]
    assert row["attempts"] == 32
    assert row["passing_yards"] == 265


def test_season_ending_team_offensive_environment_averages_a_completed_season() -> None:
    passing = pd.DataFrame(
        [
            {"season": 2022, "week": 1, "team": "BUF", "attempts": 30, "passing_yards": 300},
            {"season": 2022, "week": 2, "team": "BUF", "attempts": 40, "passing_yards": 200},
        ]
    )
    scores = pd.DataFrame(
        [
            {"season": 2022, "week": 1, "team": "BUF", "points": 20.0},
            {"season": 2022, "week": 2, "team": "BUF", "points": 30.0},
        ]
    )

    result = season_ending_team_offensive_environment(passing, scores, season=2022)

    row = result.set_index("team").loc["BUF"]
    assert row["team_pass_attempts_per_game"] == pytest.approx(35.0)
    assert row["team_passing_ypa"] == pytest.approx(500 / 70)
    assert row["team_points_per_game"] == pytest.approx(25.0)


def test_team_offensive_environment_history_uses_only_the_prior_seasons_completed_data() -> None:
    weekly = pd.DataFrame(
        [
            {"season": 2022, "week": 1, "recent_team": "BUF", "attempts": 30, "passing_yards": 300},
            # 2023's own (not-yet-known-in-advance) data must NOT be used.
            {"season": 2023, "week": 1, "recent_team": "BUF", "attempts": 999, "passing_yards": 9999},
        ]
    )
    scores = pd.DataFrame(
        [
            {"season": 2022, "week": 1, "team": "BUF", "points": 20.0},
            {"season": 2023, "week": 1, "team": "BUF", "points": 999.0},
        ]
    )

    result = team_offensive_environment_history(weekly, scores, seasons=[2022, 2023])

    row = result.set_index("team").loc["BUF"]
    assert row["team_pass_attempts_per_game"] == 30.0
    assert row["team_points_per_game"] == 20.0
    # 2022 has no prior completed season to draw from.
    assert 2022 not in set(result["season"])


def test_team_offensive_environment_by_player_joins_via_the_players_roster_team() -> None:
    history = pd.DataFrame(
        [
            {
                "season": 2023,
                "team": "BUF",
                "team_pass_attempts_per_game": 35.0,
                "team_points_per_game": 25.0,
                "team_passing_ypa": 7.5,
            }
        ]
    )
    rosters = pd.DataFrame([{"player_id": "p1", "season": 2023, "team": "BUF"}])

    result = team_offensive_environment_by_player(history, rosters)

    row = result.set_index(["season", "player_id"]).loc[(2023, "p1")]
    assert row["team_pass_attempts_per_game"] == 35.0
    assert row["team_points_per_game"] == 25.0
    assert row["team_passing_ypa"] == 7.5


def test_team_offensive_environment_by_player_deduplicates_a_player_traded_mid_season() -> None:
    history = pd.DataFrame(
        [
            {
                "season": 2023,
                "team": "BUF",
                "team_pass_attempts_per_game": 35.0,
                "team_points_per_game": 25.0,
                "team_passing_ypa": 7.5,
            },
            {
                "season": 2023,
                "team": "MIA",
                "team_pass_attempts_per_game": 40.0,
                "team_points_per_game": 30.0,
                "team_passing_ypa": 8.5,
            },
        ]
    )
    rosters = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2023, "team": "BUF"},
            {"player_id": "p1", "season": 2023, "team": "MIA"},
        ]
    )

    result = team_offensive_environment_by_player(history, rosters)

    assert len(result) == 1
