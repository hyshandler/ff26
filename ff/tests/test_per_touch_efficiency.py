import pandas as pd
import pytest

from ff_model.per_touch_efficiency import (
    prior_season_yac_above_expectation_history,
    prior_season_yards_per_target,
)


def _weekly(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _ngs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_prior_season_yards_per_target_matches_hand_computed_ratio() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "player_id": "p1", "receiving_yards": 50, "targets": 5},
            {"season": 2022, "week": 2, "player_id": "p1", "receiving_yards": 30, "targets": 3},
        ]
    )

    result = prior_season_yards_per_target(weekly, seasons=[2022, 2023])

    # 2023's feature value: 2022's totals (80 receiving yards / 8 targets) = 10.0.
    row_2023 = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row_2023["prior_season_yards_per_target"] == pytest.approx(10.0)


def test_prior_season_yards_per_target_is_absent_for_zero_prior_targets() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "player_id": "p1", "receiving_yards": 0, "targets": 0},
        ]
    )

    result = prior_season_yards_per_target(weekly, seasons=[2022, 2023])

    assert "p1" not in set(result.loc[result["season"] == 2023, "player_id"])


def test_prior_season_yac_above_expectation_shifts_season_forward() -> None:
    ngs = _ngs(
        [
            {"season": 2022, "player_id": "p1", "avg_yac_above_expectation": 1.5},
        ]
    )

    result = prior_season_yac_above_expectation_history(ngs, seasons=[2022, 2023])

    # 2023's row carries 2022's NGS value; 2022 itself has no prior-season row.
    assert set(result["season"]) == {2023}
    row = result.set_index("player_id").loc["p1"]
    assert row["prior_season_yac_above_expectation"] == pytest.approx(1.5)


def test_prior_season_yac_above_expectation_is_absent_before_ngs_coverage_begins() -> None:
    # No NGS rows at all -- as when the loader drops every season before its coverage
    # begins -- produces an empty result rather than an error.
    ngs = pd.DataFrame(columns=["season", "player_id", "avg_yac_above_expectation"])

    result = prior_season_yac_above_expectation_history(ngs, seasons=[2015])

    assert result.empty
