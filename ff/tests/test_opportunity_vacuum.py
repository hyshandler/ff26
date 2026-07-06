import pandas as pd
import pytest

from ff_model.opportunity_vacuum import (
    prior_season_points_per_target,
    vacated_target_share_history,
)


def _weekly(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _rosters(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_vacated_target_share_sums_departed_teammates_prior_share() -> None:
    weekly = _weekly(
        [
            # DAL's 2022 targets: departed=60 (60%), stays=40 (40%).
            {"season": 2022, "week": 1, "recent_team": "DAL", "player_id": "departed", "targets": 60},
            {"season": 2022, "week": 1, "recent_team": "DAL", "player_id": "stays", "targets": 40},
        ]
    )
    rosters = _rosters(
        [
            {"player_id": "departed", "season": 2022, "team": "DAL"},
            {"player_id": "stays", "season": 2022, "team": "DAL"},
            # "departed" isn't on any 2023 roster at all -- pure roster departure.
            {"player_id": "stays", "season": 2023, "team": "DAL"},
        ]
    )

    result = vacated_target_share_history(weekly, rosters, seasons=[2022, 2023])

    row = result.set_index("player_id").loc["stays"]
    assert row["vacated_target_share"] == pytest.approx(0.6)


def test_a_teammate_still_rostered_anywhere_does_not_count_as_departed() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "recent_team": "DAL", "player_id": "traded", "targets": 60},
            {"season": 2022, "week": 1, "recent_team": "DAL", "player_id": "stays", "targets": 40},
        ]
    )
    rosters = _rosters(
        [
            {"player_id": "traded", "season": 2022, "team": "DAL"},
            {"player_id": "stays", "season": 2022, "team": "DAL"},
            # "traded" moved to a new team for 2023 -- still rostered, so not vacated.
            {"player_id": "traded", "season": 2023, "team": "NYG"},
            {"player_id": "stays", "season": 2023, "team": "DAL"},
        ]
    )

    result = vacated_target_share_history(weekly, rosters, seasons=[2022, 2023])

    row = result.set_index("player_id").loc["stays"]
    assert row["vacated_target_share"] == pytest.approx(0.0)


def test_vacated_target_share_is_zero_when_no_teammates_departed() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "recent_team": "DAL", "player_id": "stays", "targets": 40},
        ]
    )
    rosters = _rosters(
        [
            {"player_id": "stays", "season": 2022, "team": "DAL"},
            {"player_id": "stays", "season": 2023, "team": "DAL"},
        ]
    )

    result = vacated_target_share_history(weekly, rosters, seasons=[2022, 2023])

    row = result.set_index("player_id").loc["stays"]
    assert row["vacated_target_share"] == pytest.approx(0.0)


def test_prior_season_points_per_target_matches_ppr_formula() -> None:
    weekly = _weekly(
        [
            # 80 receiving yards, 8 receptions, 1 TD, 20 targets:
            # PPR points = 80*0.1 + 8*1 + 1*6 = 22; points per target = 22 / 20 = 1.1.
            {"season": 2022, "week": 1, "player_id": "p1", "receiving_yards": 50, "receiving_tds": 1, "receptions": 5, "targets": 12},
            {"season": 2022, "week": 2, "player_id": "p1", "receiving_yards": 30, "receiving_tds": 0, "receptions": 3, "targets": 8},
        ]
    )

    result = prior_season_points_per_target(weekly, seasons=[2022, 2023])

    row_2023 = result.set_index("player_id").loc["p1"]
    assert row_2023["prior_season_points_per_target"] == pytest.approx(1.1)


def test_prior_season_points_per_target_absent_for_zero_targets() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "player_id": "p1", "receiving_yards": 0, "receiving_tds": 0, "receptions": 0, "targets": 0},
        ]
    )

    result = prior_season_points_per_target(weekly, seasons=[2022, 2023])

    assert "p1" not in set(result.loc[result["season"] == 2023, "player_id"])
