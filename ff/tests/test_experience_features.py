import pandas as pd

from ff_model.experience_features import (
    age_history,
    career_games_history,
    career_stage_bucket_history,
    years_in_league_history,
)


def _rosters(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _weekly(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_age_history_reads_age_directly_since_it_is_known_before_the_season_is_played() -> None:
    rosters = _rosters(
        [
            {"season": 2022, "player_id": "p1", "age": 25.0},
            {"season": 2023, "player_id": "p1", "age": 26.0},
        ]
    )

    result = age_history(rosters, seasons=[2022, 2023])

    row_2023 = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row_2023["age"] == 26.0


def test_age_history_deduplicates_a_player_traded_mid_season() -> None:
    rosters = _rosters(
        [
            {"season": 2022, "player_id": "p1", "age": 25.0, "team": "NYJ"},
            {"season": 2022, "player_id": "p1", "age": 25.0, "team": "MIA"},
        ]
    )

    result = age_history(rosters, seasons=[2022])

    assert len(result) == 1


def test_years_in_league_history_is_season_minus_rookie_year() -> None:
    rosters = _rosters(
        [
            {"season": 2023, "player_id": "p1", "rookie_year": 2020},
        ]
    )

    result = years_in_league_history(rosters, seasons=[2023])

    row = result.set_index("player_id").loc["p1"]
    assert row["years_in_league"] == 3


def test_career_stage_bucket_history_buckets_the_seasons_own_age() -> None:
    rosters = _rosters(
        [
            {"season": 2023, "player_id": "p1", "age": 28.0},
        ]
    )

    result = career_stage_bucket_history(rosters, seasons=[2023])

    row = result.set_index("player_id").loc["p1"]
    assert row["career_stage_bucket"] == "27-29"


def test_career_games_history_is_blind_to_the_current_and_later_seasons() -> None:
    weekly = _weekly(
        [{"season": 2021, "week": w, "player_id": "p1"} for w in range(1, 15)]
        + [{"season": 2022, "week": w, "player_id": "p1"} for w in range(1, 11)]
    )

    result = career_games_history(weekly, seasons=[2021, 2022, 2023])

    # 2022's value must reflect only 2021's 14 games, not 2022's own games.
    row_2022 = result.loc[result["season"] == 2022].set_index("player_id").loc["p1"]
    assert row_2022["career_games"] == 14

    # 2023's value must include both 2021 and 2022 (14 + 10 = 24).
    row_2023 = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row_2023["career_games"] == 24

    # The earliest season in `seasons` has no prior season, so it's absent.
    assert 2021 not in set(result["season"])
