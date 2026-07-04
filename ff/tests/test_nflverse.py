import pytest

from ff_model.nflverse import (
    load_draft_picks,
    load_injury_reports,
    load_offense_snap_pct,
    load_red_zone_rush_attempts,
    load_schedules,
    load_seasonal_rosters,
    load_weekly_stats,
    pfr_id_crosswalk,
)


@pytest.mark.network
def test_load_injury_reports_returns_regular_season_rows_only() -> None:
    result = load_injury_reports([2022])

    assert {"player_id", "season", "week", "report_status"} <= set(result.columns)
    assert len(result) > 0


@pytest.mark.network
def test_load_draft_picks_returns_round_and_team_per_player() -> None:
    result = load_draft_picks([2022])

    assert {"player_id", "season", "round", "position", "team"} <= set(result.columns)
    assert len(result) > 0
    assert result["round"].min() >= 1


@pytest.mark.network
def test_load_weekly_stats_excludes_postseason_and_week_18() -> None:
    result = load_weekly_stats([2021])

    assert (result["season_type"] == "REG").all()
    assert (result["week"] <= 17).all()


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


@pytest.mark.network
def test_load_schedules_gives_each_team_one_row_per_week_with_its_opponent() -> None:
    result = load_schedules([2023])

    assert {"season", "week", "team", "opponent_team"} <= set(result.columns)
    assert (result["week"] <= 17).all()
    # Every team plays each other team, never itself.
    assert (result["team"] != result["opponent_team"]).all()
    # 32 teams x ~17 games (minus byes) each -- one row per team per game played.
    kc_week1 = result.loc[(result["team"] == "KC") & (result["week"] == 1) & (result["season"] == 2023)]
    assert len(kc_week1) == 1
    assert kc_week1.iloc[0]["opponent_team"] == "DET"
