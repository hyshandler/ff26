import pandas as pd
import pytest

from ff_model.naive_baseline import RAW_STAT_COLUMNS
from ff_model.rb_model import FEATURE_COLUMNS, add_rb_features, build_rb_projections


def _weekly(rows: list[dict]) -> pd.DataFrame:
    base = {
        "carries": 0,
        "rushing_yards": 0,
        "rushing_tds": 0,
        "targets": 0,
        "receptions": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "position": "RB",
        "recent_team": "DAL",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_add_rb_features_only_returns_rb_rows() -> None:
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "player_id": "rb1", "position": "RB", "carries": 10},
            {"season": 2022, "week": 1, "player_id": "qb1", "position": "QB", "carries": 2},
        ]
    )
    red_zone = pd.DataFrame(columns=["season", "week", "player_id", "red_zone_carries"])
    snap_pct = pd.DataFrame(columns=["season", "week", "player_id", "offense_pct"])

    result = add_rb_features(weekly, red_zone, snap_pct)

    assert set(result["player_id"]) == {"rb1"}
    assert set(FEATURE_COLUMNS) <= set(result.columns)
    assert set(RAW_STAT_COLUMNS) <= set(result.columns)


def test_add_rb_features_uses_all_positions_for_team_totals() -> None:
    # rb1 has all 10 of the team's carries in week 1 among RBs, but the QB's 2 scrambles
    # count toward the team total a real carry-share feature must divide by.
    weekly = _weekly(
        [
            {"season": 2022, "week": 1, "player_id": "rb1", "position": "RB", "carries": 10},
            {"season": 2022, "week": 1, "player_id": "qb1", "position": "QB", "carries": 2},
            {"season": 2022, "week": 2, "player_id": "rb1", "position": "RB", "carries": 5},
        ]
    )
    red_zone = pd.DataFrame(columns=["season", "week", "player_id", "red_zone_carries"])
    snap_pct = pd.DataFrame(columns=["season", "week", "player_id", "offense_pct"])

    result = add_rb_features(weekly, red_zone, snap_pct)

    week2 = result.loc[result["week"] == 2].iloc[0]
    assert week2["carry_share"] == 10 / 12


@pytest.mark.network
def test_build_rb_projections_produces_monotonic_quantiles_for_each_raw_stat() -> None:
    from ff_model.nflverse import (
        load_offense_snap_pct,
        load_red_zone_rush_attempts,
        load_seasonal_rosters,
        load_weekly_stats,
        pfr_id_crosswalk,
    )

    seasons = list(range(2018, 2023))
    weekly = load_weekly_stats(seasons)
    rosters = load_seasonal_rosters(seasons)
    red_zone = load_red_zone_rush_attempts(seasons)

    pfr_id_by_player_id = pfr_id_crosswalk(rosters)
    snap_pct = load_offense_snap_pct(seasons, pfr_id_by_player_id)

    rb_ids = set(
        weekly.loc[(weekly["position"] == "RB") & (weekly["season"] == 2022), "player_id"]
    )
    sample_ids = set(list(rb_ids)[:15])

    result = build_rb_projections(
        weekly,
        red_zone,
        snap_pct,
        train_through_season=2022,
        target_season=2023,
        eligible_player_ids=sample_ids,
    )

    assert len(result) > 0
    for stat in RAW_STAT_COLUMNS:
        assert (result[f"{stat}_p10"] <= result[f"{stat}_p50"]).all()
        assert (result[f"{stat}_p50"] <= result[f"{stat}_p90"]).all()
