import pandas as pd
import pytest

from ff_model.features import (
    add_trailing_player_averages,
    add_trailing_team_shares,
    multi_season_career_averages,
    multi_season_last_n_averages,
    multi_season_recency_weighted_averages,
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


def test_multi_season_career_average_is_blind_to_the_current_and_later_seasons() -> None:
    weekly = _weekly(
        [
            {"season": 2021, "week": 1, "player_id": "p1", "rushing_yards": 10},
            {"season": 2021, "week": 2, "player_id": "p1", "rushing_yards": 30},
            {"season": 2022, "week": 1, "player_id": "p1", "rushing_yards": 999},
        ]
    )

    result = multi_season_career_averages(
        weekly, {"career_avg_rushing_yards": "rushing_yards"}, seasons=[2021, 2022, 2023]
    )

    # 2022's feature value must reflect only 2021 (20 yds/game over 2 games), not 2022 itself.
    row_2022 = result.loc[result["season"] == 2022].set_index("player_id").loc["p1"]
    assert row_2022["career_avg_rushing_yards"] == 20.0

    # 2023's feature value must be the career average across both 2021 and 2022 so far
    # (2021: 10+30=40 over 2 games, 2022: 999 over 1 game -> 1039/3).
    row_2023 = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row_2023["career_avg_rushing_yards"] == pytest.approx(1039 / 3)

    # The earliest season in `seasons` has no prior season, so it's absent from the result.
    assert 2021 not in set(result["season"])


def test_multi_season_last_n_averages_only_uses_the_most_recent_n_seasons() -> None:
    weekly = _weekly(
        [
            {"season": 2020, "week": 1, "player_id": "p1", "rushing_yards": 9999},
            {"season": 2021, "week": 1, "player_id": "p1", "rushing_yards": 10},
            {"season": 2021, "week": 2, "player_id": "p1", "rushing_yards": 30},
            {"season": 2022, "week": 1, "player_id": "p1", "rushing_yards": 100},
        ]
    )

    result = multi_season_last_n_averages(
        weekly,
        {"last2_avg_rushing_yards": "rushing_yards"},
        seasons=[2020, 2021, 2022, 2023],
        n_seasons=2,
    )

    # 2023's value should use only 2021+2022 (the last 2 prior seasons), excluding 2020's
    # huge outlier: (10 + 30 + 100) / 3 games.
    row = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row["last2_avg_rushing_yards"] == pytest.approx(140 / 3)


def test_multi_season_recency_weighted_average_weights_more_recent_seasons_higher() -> None:
    weekly = _weekly(
        [
            {"season": 2021, "week": 1, "player_id": "p1", "rushing_yards": 10},
            {"season": 2022, "week": 1, "player_id": "p1", "rushing_yards": 20},
        ]
    )

    result = multi_season_recency_weighted_averages(
        weekly,
        {"ewm_avg_rushing_yards": "rushing_yards"},
        seasons=[2021, 2022, 2023],
        decay=0.5,
    )

    # 2023 is 2 seasons after 2021 (weight 0.5) and 1 season after 2022 (weight 1.0):
    # (10*0.5 + 20*1.0) / (0.5 + 1.0) = 25/1.5
    row = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row["ewm_avg_rushing_yards"] == pytest.approx(25 / 1.5)
