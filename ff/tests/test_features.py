import pandas as pd

from ff_model.features import (
    add_trailing_player_averages,
    add_trailing_team_shares,
    season_ending_averages,
    season_ending_shares,
)


def _weekly(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_trailing_share_is_blind_to_the_current_and_later_weeks() -> None:
    weekly = _weekly(
        [
            {"season": 2023, "week": 1, "recent_team": "DAL", "player_id": "p1", "carries": 10},
            {"season": 2023, "week": 1, "recent_team": "DAL", "player_id": "p2", "carries": 10},
            {"season": 2023, "week": 2, "recent_team": "DAL", "player_id": "p1", "carries": 100},
            {"season": 2023, "week": 2, "recent_team": "DAL", "player_id": "p2", "carries": 0},
            {"season": 2023, "week": 3, "recent_team": "DAL", "player_id": "p1", "carries": 15},
            {"season": 2023, "week": 3, "recent_team": "DAL", "player_id": "p2", "carries": 5},
        ]
    )

    result = add_trailing_team_shares(weekly, {"carry_share": "carries"})

    week1 = result.loc[(result["week"] == 1) & (result["player_id"] == "p1"), "carry_share"]
    assert week1.isna().all()  # no prior weeks this season yet

    week2 = result.loc[(result["week"] == 2) & (result["player_id"] == "p1"), "carry_share"].item()
    assert week2 == 0.5  # week 1 only: p1 had 10 of the team's 20 carries

    # Week 3's huge week-2 carries for p1 must be reflected in week 3's trailing share,
    # not leak backward into week 2's feature value computed above.
    week3 = result.loc[(result["week"] == 3) & (result["player_id"] == "p1"), "carry_share"].item()
    assert week3 == (10 + 100) / (20 + 100)


def test_trailing_share_resets_at_a_season_boundary() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 17, "recent_team": "DAL", "player_id": "p1", "carries": 999},
            {"season": 2022, "week": 17, "recent_team": "DAL", "player_id": "p2", "carries": 1},
            {"season": 2023, "week": 1, "recent_team": "DAL", "player_id": "p1", "carries": 10},
            {"season": 2023, "week": 1, "recent_team": "DAL", "player_id": "p2", "carries": 10},
        ]
    )

    result = add_trailing_team_shares(weekly, {"carry_share": "carries"})

    week1_2023 = result.loc[
        (result["season"] == 2023) & (result["player_id"] == "p1"), "carry_share"
    ]
    assert week1_2023.isna().all()


def test_season_ending_shares_use_the_whole_completed_season() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "recent_team": "DAL", "player_id": "p1", "carries": 10},
            {"season": 2022, "week": 2, "recent_team": "DAL", "player_id": "p1", "carries": 10},
            {"season": 2022, "week": 1, "recent_team": "DAL", "player_id": "p2", "carries": 5},
            {"season": 2022, "week": 2, "recent_team": "DAL", "player_id": "p2", "carries": 5},
        ]
    )

    result = season_ending_shares(weekly, {"carry_share": "carries"}, season=2022)

    row = result.set_index("player_id").loc["p1"]
    assert row["carry_share"] == 20 / 30


def test_trailing_player_average_is_blind_to_the_current_and_later_weeks() -> None:
    weekly = _weekly(
        [
            {"season": 2023, "week": 1, "player_id": "p1", "rushing_yards": 10},
            {"season": 2023, "week": 2, "player_id": "p1", "rushing_yards": 30},
            {"season": 2023, "week": 3, "player_id": "p1", "rushing_yards": 999},
        ]
    )

    result = add_trailing_player_averages(weekly, {"trailing_avg_rushing_yards": "rushing_yards"})

    week1 = result.loc[result["week"] == 1, "trailing_avg_rushing_yards"]
    assert week1.isna().all()

    week2 = result.loc[result["week"] == 2, "trailing_avg_rushing_yards"].item()
    assert week2 == 10.0

    week3 = result.loc[result["week"] == 3, "trailing_avg_rushing_yards"].item()
    assert week3 == 20.0


def test_season_ending_averages_use_the_whole_completed_season() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "player_id": "p1", "rushing_yards": 10},
            {"season": 2022, "week": 2, "player_id": "p1", "rushing_yards": 30},
        ]
    )

    result = season_ending_averages(
        weekly, {"trailing_avg_rushing_yards": "rushing_yards"}, season=2022
    )

    row = result.set_index("player_id").loc["p1"]
    assert row["trailing_avg_rushing_yards"] == 20.0
