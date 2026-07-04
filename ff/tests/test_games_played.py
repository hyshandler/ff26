import pandas as pd

from ff_model.games_played import estimate_games_played


def _rosters(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _weekly(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


EMPTY_INJURIES = pd.DataFrame(columns=["player_id", "season", "week", "report_status"])


def test_base_rate_matches_the_cohorts_average_games_played() -> None:
    rosters = _rosters(
        [
            {"player_id": "a", "season": 2021, "age": 25.0, "position": "RB"},
            {"player_id": "b", "season": 2021, "age": 25.0, "position": "RB"},
            {"player_id": "target", "season": 2022, "age": 25.0, "position": "RB"},
        ]
    )
    weekly = _weekly(
        [{"player_id": "a", "season": 2021, "week": w, "position": "RB"} for w in range(1, 15)]
        + [{"player_id": "b", "season": 2021, "week": w, "position": "RB"} for w in range(1, 11)]
    )

    result = estimate_games_played(
        rosters,
        weekly,
        EMPTY_INJURIES,
        train_through_season=2021,
        target_season=2022,
        player_ids={"target"},
    )

    # Cohort (RB, age 25) played 14 and 10 games respectively -> average 12.
    assert result["target"] == 12.0


def test_injury_flagged_players_use_the_flagged_cohorts_average() -> None:
    rosters = _rosters(
        [
            {"player_id": "healthy_hist", "season": 2021, "age": 25.0, "position": "RB"},
            {"player_id": "flagged_hist", "season": 2021, "age": 25.0, "position": "RB"},
            {"player_id": "target", "season": 2022, "age": 25.0, "position": "RB"},
        ]
    )
    weekly = _weekly(
        [
            {"player_id": "healthy_hist", "season": 2021, "week": w, "position": "RB"}
            for w in range(1, 18)
        ]
        + [
            {"player_id": "flagged_hist", "season": 2021, "week": w, "position": "RB"}
            for w in range(1, 11)
        ]
    )
    # Both `flagged_hist` (builds the flagged cohort's average) and `target` (the
    # player being projected) were reported "Out" in the final 3 weeks of train_through_season.
    injuries = pd.DataFrame(
        [
            {"player_id": player_id, "season": 2021, "week": w, "report_status": "Out"}
            for player_id in ("flagged_hist", "target")
            for w in (15, 16, 17)
        ]
    )

    result = estimate_games_played(
        rosters,
        weekly,
        injuries,
        train_through_season=2021,
        target_season=2022,
        player_ids={"target"},
    )

    # target is flagged -> uses the flagged cohort's average (flagged_hist's 10 games),
    # not the unflagged cohort's average (healthy_hist's 17 games).
    assert result["target"] == 10.0


def test_a_player_traded_mid_season_with_duplicate_roster_rows_still_gets_one_estimate() -> None:
    """nflverse's seasonal rosters can list a player once per team in a season they
    changed teams -- e.g. two rows for `target` in 2022, one per team -- which must not
    produce a non-unique index (map()-ing this Series onto predictions would blow up)."""
    rosters = _rosters(
        [
            {"player_id": "a", "season": 2021, "age": 25.0, "position": "RB"},
            {"player_id": "target", "season": 2022, "age": 25.0, "position": "RB", "team": "NYJ"},
            {"player_id": "target", "season": 2022, "age": 25.0, "position": "RB", "team": "MIA"},
        ]
    )
    weekly = _weekly(
        [{"player_id": "a", "season": 2021, "week": w, "position": "RB"} for w in range(1, 13)]
    )

    result = estimate_games_played(
        rosters,
        weekly,
        EMPTY_INJURIES,
        train_through_season=2021,
        target_season=2022,
        player_ids={"target"},
    )

    assert result.index.is_unique
    assert result["target"] == 12.0
