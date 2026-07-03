import pandas as pd
import pytest

from ff_model.position_config import POSITION_CONFIGS, PositionConfig
from ff_model.position_model import (
    add_position_features,
    build_position_model_projections,
    feature_columns,
)

EMPTY_RED_ZONE = pd.DataFrame(columns=["season", "week", "player_id", "red_zone_carries"])
EMPTY_SNAP_PCT = pd.DataFrame(columns=["season", "week", "player_id", "offense_pct"])


def _weekly(position: str, config: PositionConfig, rows: list[dict]) -> pd.DataFrame:
    base = {stat: 0 for stat in config.raw_stat_columns} | {
        "receiving_air_yards": 0,
        "position": position,
        "recent_team": "DAL",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


@pytest.mark.parametrize("position", list(POSITION_CONFIGS))
def test_add_position_features_only_returns_that_positions_rows(position: str) -> None:
    config = POSITION_CONFIGS[position]
    weekly = _weekly(
        position,
        config,
        [
            {"season": 2022, "week": 1, "player_id": "p1", "position": position},
            {"season": 2022, "week": 1, "player_id": "other", "position": "K"},
        ],
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT)

    assert set(result["player_id"]) == {"p1"}
    assert set(feature_columns(config)) <= set(result.columns)
    assert set(config.raw_stat_columns) <= set(result.columns)


@pytest.mark.parametrize("position", list(POSITION_CONFIGS))
def test_position_share_feature_is_blind_to_the_current_and_later_weeks(position: str) -> None:
    config = POSITION_CONFIGS[position]
    feature_name, stat_column = next(iter(config.share_stat_columns.items()))

    weekly = _weekly(
        position,
        config,
        [
            {"season": 2022, "week": 1, "player_id": "p1", stat_column: 10},
            {"season": 2022, "week": 1, "player_id": "other", stat_column: 2},
            {"season": 2022, "week": 2, "player_id": "p1", stat_column: 5},
        ],
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT)

    week1 = result.loc[result["week"] == 1, feature_name]
    assert week1.isna().all()  # no prior weeks this season yet

    week2 = result.loc[result["week"] == 2].iloc[0]
    assert week2[feature_name] == 10 / 12


@pytest.mark.parametrize("position", list(POSITION_CONFIGS))
def test_position_share_feature_uses_all_positions_for_team_totals(position: str) -> None:
    config = POSITION_CONFIGS[position]
    feature_name, stat_column = next(iter(config.share_stat_columns.items()))
    other_position = "QB" if position != "QB" else "RB"

    weekly = pd.DataFrame(
        [
            {
                **{stat: 0 for stat in config.raw_stat_columns},
                "receiving_air_yards": 0,
                "recent_team": "DAL",
                "season": 2022,
                "week": 1,
                "player_id": "p1",
                "position": position,
                stat_column: 10,
            },
            {
                **{stat: 0 for stat in config.raw_stat_columns},
                "receiving_air_yards": 0,
                "recent_team": "DAL",
                "season": 2022,
                "week": 1,
                "player_id": "teammate",
                "position": other_position,
                stat_column: 2,
            },
            {
                **{stat: 0 for stat in config.raw_stat_columns},
                "receiving_air_yards": 0,
                "recent_team": "DAL",
                "season": 2022,
                "week": 2,
                "player_id": "p1",
                "position": position,
            },
        ]
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT)

    week2 = result.loc[result["week"] == 2].iloc[0]
    # A teammate at a different position must still count toward the team total
    # this position's share feature divides by.
    assert week2[feature_name] == 10 / 12


@pytest.mark.network
def test_build_position_model_projections_produces_monotonic_quantiles_for_every_position() -> (
    None
):
    from ff_model.nflverse import (
        load_offense_snap_pct,
        load_red_zone_rush_attempts,
        load_seasonal_rosters,
        load_weekly_stats,
        pfr_id_crosswalk,
    )
    from ff_model.position_model import build_position_model_projections

    seasons = list(range(2018, 2023))
    weekly = load_weekly_stats(seasons)
    rosters = load_seasonal_rosters(seasons)
    pfr_id_by_player_id = pfr_id_crosswalk(rosters)
    snap_pct = load_offense_snap_pct(seasons, pfr_id_by_player_id)
    red_zone = load_red_zone_rush_attempts(seasons)

    for position, config in POSITION_CONFIGS.items():
        player_ids = set(
            weekly.loc[(weekly["position"] == position) & (weekly["season"] == 2022), "player_id"]
        )
        sample_ids = set(list(player_ids)[:15])

        result = build_position_model_projections(
            config,
            weekly,
            red_zone,
            snap_pct,
            train_through_season=2022,
            target_season=2023,
            eligible_player_ids=sample_ids,
        )

        assert len(result) > 0, position
        for stat in config.raw_stat_columns:
            assert (result[f"{stat}_p10"] <= result[f"{stat}_p50"]).all(), (position, stat)
            assert (result[f"{stat}_p50"] <= result[f"{stat}_p90"]).all(), (position, stat)
