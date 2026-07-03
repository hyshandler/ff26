import pytest

from ff_model.nflverse import (
    load_offense_snap_pct,
    load_red_zone_rush_attempts,
    load_seasonal_rosters,
    pfr_id_crosswalk,
)


@pytest.mark.network
def test_load_red_zone_rush_attempts_returns_per_player_week_counts() -> None:
    result = load_red_zone_rush_attempts([2023])

    assert {"season", "week", "player_id", "red_zone_carries"} <= set(result.columns)
    assert (result["red_zone_carries"] > 0).all()
    assert len(result) > 0


@pytest.mark.network
def test_load_offense_snap_pct_crosswalks_to_gsis_player_id() -> None:
    rosters = load_seasonal_rosters([2023])
    pfr_id_by_player_id = pfr_id_crosswalk(rosters)

    result = load_offense_snap_pct([2023], pfr_id_by_player_id)

    assert {"player_id", "season", "week", "offense_pct"} <= set(result.columns)
    assert len(result) > 0
    assert result["offense_pct"].between(0, 1).all()
