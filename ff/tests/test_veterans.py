import pandas as pd

from ff_model.veterans import veteran_player_ids


def _rosters(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _weekly(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_excludes_rookie_in_the_target_season() -> None:
    rosters = _rosters(
        [{"player_id": "rookie-1", "season": 2023, "rookie_year": 2023, "position": "RB"}]
    )
    weekly = pd.DataFrame(columns=["player_id", "season", "week", "position"])

    result = veteran_player_ids(rosters, weekly, season=2023, min_career_games=1)

    assert "rookie-1" not in result


def test_includes_a_player_past_the_rookie_season_with_enough_career_games() -> None:
    rosters = _rosters(
        [{"player_id": "vet-1", "season": 2023, "rookie_year": 2021, "position": "RB"}]
    )
    weekly = _weekly(
        [
            {"player_id": "vet-1", "season": 2021, "week": w, "position": "RB"}
            for w in range(1, 17)
        ]
    )

    result = veteran_player_ids(rosters, weekly, season=2023, min_career_games=16)

    assert "vet-1" in result


def test_excludes_a_second_year_player_below_the_career_games_threshold() -> None:
    rosters = _rosters(
        [{"player_id": "sparse-1", "season": 2023, "rookie_year": 2022, "position": "RB"}]
    )
    weekly = _weekly(
        [{"player_id": "sparse-1", "season": 2022, "week": 5, "position": "RB"}]
    )

    result = veteran_player_ids(rosters, weekly, season=2023, min_career_games=16)

    assert "sparse-1" not in result


def test_only_counts_games_strictly_before_the_target_season() -> None:
    rosters = _rosters(
        [{"player_id": "vet-2", "season": 2023, "rookie_year": 2021, "position": "RB"}]
    )
    weekly = _weekly(
        [{"player_id": "vet-2", "season": 2021, "week": 1, "position": "RB"}]
        + [
            {"player_id": "vet-2", "season": 2023, "week": w, "position": "RB"}
            for w in range(1, 16)
        ]
    )

    result = veteran_player_ids(rosters, weekly, season=2023, min_career_games=16)

    assert "vet-2" not in result
