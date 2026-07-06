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
EMPTY_DEPTH_CHART = pd.DataFrame(columns=["season", "player_id", "depth_chart_competition"])


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

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, EMPTY_DEPTH_CHART)

    assert set(result["player_id"]) == {"p1"}
    assert set(feature_columns(config)) <= set(result.columns)
    assert set(config.raw_stat_columns) <= set(result.columns)


def test_feature_columns_can_exclude_depth_chart_competition_for_the_without_feature_baseline() -> None:
    config = POSITION_CONFIGS["RB"]

    assert "depth_chart_competition" in feature_columns(config)
    assert "depth_chart_competition" not in feature_columns(
        config, include_depth_chart_competition=False
    )


@pytest.mark.parametrize("position", list(POSITION_CONFIGS))
def test_add_position_features_merges_the_depth_chart_competition_flag(position: str) -> None:
    config = POSITION_CONFIGS[position]
    weekly = _weekly(
        position,
        config,
        [{"season": 2022, "week": 1, "player_id": "p1", "position": position}],
    )
    depth_chart = pd.DataFrame(
        [{"season": 2022, "player_id": "p1", "depth_chart_competition": 1}]
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, depth_chart)

    assert "depth_chart_competition" in feature_columns(config)
    assert result.set_index("player_id").loc["p1", "depth_chart_competition"] == 1


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

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, EMPTY_DEPTH_CHART)

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

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, EMPTY_DEPTH_CHART)

    week2 = result.loc[result["week"] == 2].iloc[0]
    # A teammate at a different position must still count toward the team total
    # this position's share feature divides by.
    assert week2[feature_name] == 10 / 12


def test_feature_columns_can_include_multi_season_averages() -> None:
    config = POSITION_CONFIGS["RB"]

    assert "multi_season_avg_carries" not in feature_columns(config)
    assert "multi_season_avg_carries" in feature_columns(config, include_multi_season=True)


def test_feature_columns_only_includes_prior_season_totals_for_wr() -> None:
    assert "prior_season_fantasy_points" in feature_columns(POSITION_CONFIGS["WR"])
    assert "prior_season_games_played" in feature_columns(POSITION_CONFIGS["WR"])
    for position in ("RB", "QB", "TE"):
        assert "prior_season_fantasy_points" not in feature_columns(POSITION_CONFIGS[position])
        assert "prior_season_games_played" not in feature_columns(POSITION_CONFIGS[position])


def test_add_position_features_computes_prior_season_totals_for_wr() -> None:
    config = POSITION_CONFIGS["WR"]
    weekly = _weekly(
        "WR",
        config,
        [
            {
                "season": 2022,
                "week": 1,
                "player_id": "p1",
                "receiving_yards": 100,
                "receptions": 10,
                "receiving_tds": 1,
            },
            {"season": 2023, "week": 1, "player_id": "p1"},
        ],
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, EMPTY_DEPTH_CHART)

    row_2023 = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row_2023["prior_season_fantasy_points"] == pytest.approx(100 * 0.1 + 10 * 1 + 1 * 6)
    assert row_2023["prior_season_games_played"] == 1

    # 2022 is p1's first season -- no prior season, so NaN, not zero.
    row_2022 = result.loc[result["season"] == 2022].set_index("player_id").loc["p1"]
    assert pd.isna(row_2022["prior_season_fantasy_points"])
    assert pd.isna(row_2022["prior_season_games_played"])


@pytest.mark.parametrize("position", ["RB", "QB", "TE"])
def test_add_position_features_does_not_add_prior_season_totals_for_other_positions(
    position: str,
) -> None:
    config = POSITION_CONFIGS[position]
    weekly = _weekly(
        position, config, [{"season": 2022, "week": 1, "player_id": "p1", "position": position}]
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, EMPTY_DEPTH_CHART)

    assert "prior_season_fantasy_points" not in result.columns
    assert "prior_season_games_played" not in result.columns


def test_feature_columns_only_includes_opportunity_vacuum_for_wr() -> None:
    assert "vacated_target_share" in feature_columns(POSITION_CONFIGS["WR"])
    assert "prior_season_points_per_target" in feature_columns(POSITION_CONFIGS["WR"])
    for position in ("RB", "QB", "TE"):
        assert "vacated_target_share" not in feature_columns(POSITION_CONFIGS[position])
        assert "prior_season_points_per_target" not in feature_columns(POSITION_CONFIGS[position])


def test_add_position_features_computes_opportunity_vacuum_for_wr() -> None:
    config = POSITION_CONFIGS["WR"]
    weekly = _weekly(
        "WR",
        config,
        [
            {
                "season": 2022,
                "week": 1,
                "player_id": "p1",
                "receiving_yards": 100,
                "receptions": 10,
                "receiving_tds": 1,
                "targets": 20,
            },
            {"season": 2023, "week": 1, "player_id": "p1", "targets": 5},
        ],
    )
    opportunity_vacuum_history = pd.DataFrame(
        [
            {"season": 2023, "player_id": "p1", "vacated_target_share": 0.25},
        ]
    )

    result = add_position_features(
        config,
        weekly,
        EMPTY_RED_ZONE,
        EMPTY_SNAP_PCT,
        EMPTY_DEPTH_CHART,
        opportunity_vacuum_history=opportunity_vacuum_history,
    )

    row_2023 = result.loc[result["season"] == 2023].set_index("player_id").loc["p1"]
    assert row_2023["vacated_target_share"] == pytest.approx(0.25)
    assert row_2023["prior_season_points_per_target"] == pytest.approx(
        (100 * 0.1 + 10 * 1 + 1 * 6) / 20
    )

    # 2022 has no prior season and no vacated-share row -- NaN own-efficiency, 0 vacancy.
    row_2022 = result.loc[result["season"] == 2022].set_index("player_id").loc["p1"]
    assert pd.isna(row_2022["prior_season_points_per_target"])
    assert row_2022["vacated_target_share"] == pytest.approx(0.0)


@pytest.mark.parametrize("position", ["RB", "QB", "TE"])
def test_add_position_features_does_not_add_opportunity_vacuum_for_other_positions(
    position: str,
) -> None:
    config = POSITION_CONFIGS[position]
    weekly = _weekly(
        position, config, [{"season": 2022, "week": 1, "player_id": "p1", "position": position}]
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, EMPTY_DEPTH_CHART)

    assert "vacated_target_share" not in result.columns
    assert "prior_season_points_per_target" not in result.columns


def test_feature_columns_only_includes_team_offensive_environment_for_wr() -> None:
    team_env_columns = ["team_pass_attempts_per_game", "team_points_per_game", "team_passing_ypa"]
    for column in team_env_columns:
        assert column in feature_columns(POSITION_CONFIGS["WR"])
    for position in ("RB", "QB", "TE"):
        for column in team_env_columns:
            assert column not in feature_columns(POSITION_CONFIGS[position])


def test_add_position_features_computes_team_offensive_environment_for_wr() -> None:
    config = POSITION_CONFIGS["WR"]
    weekly = _weekly(
        "WR",
        config,
        [
            {"season": 2023, "week": 1, "player_id": "p1"},
        ],
    )
    team_offensive_environment_history = pd.DataFrame(
        [
            {
                "season": 2023,
                "player_id": "p1",
                "team_pass_attempts_per_game": 35.0,
                "team_points_per_game": 25.0,
                "team_passing_ypa": 7.5,
            },
        ]
    )

    result = add_position_features(
        config,
        weekly,
        EMPTY_RED_ZONE,
        EMPTY_SNAP_PCT,
        EMPTY_DEPTH_CHART,
        team_offensive_environment_history=team_offensive_environment_history,
    )

    row = result.set_index("player_id").loc["p1"]
    assert row["team_pass_attempts_per_game"] == pytest.approx(35.0)
    assert row["team_points_per_game"] == pytest.approx(25.0)
    assert row["team_passing_ypa"] == pytest.approx(7.5)


@pytest.mark.parametrize("position", ["RB", "QB", "TE"])
def test_add_position_features_does_not_add_team_offensive_environment_for_other_positions(
    position: str,
) -> None:
    config = POSITION_CONFIGS[position]
    weekly = _weekly(
        position, config, [{"season": 2022, "week": 1, "player_id": "p1", "position": position}]
    )

    result = add_position_features(config, weekly, EMPTY_RED_ZONE, EMPTY_SNAP_PCT, EMPTY_DEPTH_CHART)

    assert "team_pass_attempts_per_game" not in result.columns
    assert "team_points_per_game" not in result.columns
    assert "team_passing_ypa" not in result.columns


@pytest.mark.parametrize(
    "sos_feature,expected_column", [("season_wide", "season_wide_sos"), ("actual_games", "trailing_sos_faced")]
)
def test_feature_columns_can_include_a_sos_feature(sos_feature: str, expected_column: str) -> None:
    config = POSITION_CONFIGS["RB"]

    assert expected_column not in feature_columns(config)
    assert expected_column in feature_columns(config, sos_feature=sos_feature)


@pytest.mark.parametrize(
    "experience_feature,expected_column",
    [
        ("age", "age"),
        ("years_in_league", "years_in_league"),
        ("career_games", "career_games"),
        ("career_stage_bucket", "career_stage_bucket"),
    ],
)
def test_feature_columns_can_include_an_experience_feature(
    experience_feature: str, expected_column: str
) -> None:
    config = POSITION_CONFIGS["RB"]

    assert expected_column not in feature_columns(config)
    assert expected_column in feature_columns(config, experience_feature=experience_feature)


@pytest.mark.parametrize("position", list(POSITION_CONFIGS))
def test_add_position_features_merges_the_experience_history(position: str) -> None:
    config = POSITION_CONFIGS[position]
    weekly = _weekly(
        position,
        config,
        [{"season": 2022, "week": 1, "player_id": "p1", "position": position}],
    )
    experience_history = pd.DataFrame([{"season": 2022, "player_id": "p1", "age": 27.0}])

    result = add_position_features(
        config,
        weekly,
        EMPTY_RED_ZONE,
        EMPTY_SNAP_PCT,
        EMPTY_DEPTH_CHART,
        experience_history=experience_history,
    )

    assert result.set_index("player_id").loc["p1", "age"] == 27.0


def test_add_position_features_merges_season_wide_sos_history() -> None:
    config = POSITION_CONFIGS["RB"]
    weekly = _weekly(
        "RB",
        config,
        [{"season": 2022, "week": 1, "player_id": "p1"}],
    )
    sos_history = pd.DataFrame([{"season": 2022, "player_id": "p1", "season_wide_sos": 12.5}])

    result = add_position_features(
        config,
        weekly,
        EMPTY_RED_ZONE,
        EMPTY_SNAP_PCT,
        EMPTY_DEPTH_CHART,
        sos_history=sos_history,
        sos_feature="season_wide",
    )

    assert result.set_index("player_id").loc["p1", "season_wide_sos"] == 12.5


def test_add_position_features_computes_actual_games_sos_from_each_players_own_matchups() -> None:
    config = POSITION_CONFIGS["RB"]
    weekly = _weekly(
        "RB",
        config,
        [
            {"season": 2023, "week": 1, "player_id": "p1", "opponent_team": "NYJ"},
            {"season": 2023, "week": 2, "player_id": "p1", "opponent_team": "MIA"},
        ],
    )
    league_wide_trailing_points_allowed = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "team": "NYJ", "position": "RB", "trailing_points_allowed": 10.0},
            {"season": 2023, "week": 2, "team": "MIA", "position": "RB", "trailing_points_allowed": 30.0},
        ]
    )

    result = add_position_features(
        config,
        weekly,
        EMPTY_RED_ZONE,
        EMPTY_SNAP_PCT,
        EMPTY_DEPTH_CHART,
        league_wide_trailing_points_allowed=league_wide_trailing_points_allowed,
        sos_feature="actual_games",
    )

    week1 = result.loc[result["week"] == 1, "trailing_sos_faced"]
    assert week1.isna().all()  # no prior actual games yet this season

    week2 = result.loc[result["week"] == 2, "trailing_sos_faced"].item()
    assert week2 == 10.0  # trailing average of week 1's matchup (NYJ, 10.0) only


@pytest.mark.parametrize("position", list(POSITION_CONFIGS))
def test_add_position_features_merges_the_multi_season_history(position: str) -> None:
    config = POSITION_CONFIGS[position]
    stat = config.raw_stat_columns[0]
    weekly = _weekly(
        position,
        config,
        [{"season": 2022, "week": 1, "player_id": "p1", "position": position}],
    )
    multi_season_history = pd.DataFrame(
        [{"season": 2022, "player_id": "p1", f"multi_season_avg_{stat}": 12.5}]
    )

    result = add_position_features(
        config,
        weekly,
        EMPTY_RED_ZONE,
        EMPTY_SNAP_PCT,
        EMPTY_DEPTH_CHART,
        multi_season_history=multi_season_history,
    )

    assert result.set_index("player_id").loc["p1", f"multi_season_avg_{stat}"] == 12.5


@pytest.mark.network
def test_build_position_model_projections_supports_tabfm_backend() -> None:
    config = POSITION_CONFIGS["RB"]
    rng_rows = [
        {
            "season": 2020 + (i % 3),
            "week": 1 + (i % 10),
            "player_id": f"p{i % 5}",
            "carries": float(i % 20),
            "rushing_yards": float((i % 20) * 4),
        }
        for i in range(40)
    ]
    weekly = _weekly("RB", config, rng_rows)

    result = build_position_model_projections(
        config,
        weekly,
        EMPTY_RED_ZONE,
        EMPTY_SNAP_PCT,
        EMPTY_DEPTH_CHART,
        train_through_season=2022,
        target_season=2023,
        eligible_player_ids={"p0", "p1", "p2", "p3", "p4"},
        model_backend="tabfm",
    )

    assert len(result) > 0
    for stat in config.raw_stat_columns:
        assert (result[f"{stat}_p10"] <= result[f"{stat}_p50"]).all(), stat
        assert (result[f"{stat}_p50"] <= result[f"{stat}_p90"]).all(), stat


@pytest.mark.network
def test_build_position_model_projections_produces_monotonic_quantiles_for_every_position() -> (
    None
):
    from ff_model.depth_chart import depth_chart_competition_history
    from ff_model.nflverse import (
        load_draft_picks,
        load_offense_snap_pct,
        load_red_zone_rush_attempts,
        load_seasonal_rosters,
        load_weekly_stats,
        pfr_id_crosswalk,
    )
    from ff_model.position_model import build_position_model_projections

    seasons = list(range(2018, 2023))
    weekly = load_weekly_stats(seasons)
    rosters = load_seasonal_rosters(seasons + [2023])
    pfr_id_by_player_id = pfr_id_crosswalk(rosters)
    snap_pct = load_offense_snap_pct(seasons, pfr_id_by_player_id)
    red_zone = load_red_zone_rush_attempts(seasons)
    draft_picks = load_draft_picks(seasons + [2023])
    depth_chart_history = depth_chart_competition_history(rosters, draft_picks, seasons + [2023])

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
            depth_chart_history,
            train_through_season=2022,
            target_season=2023,
            eligible_player_ids=sample_ids,
        )

        assert len(result) > 0, position
        for stat in config.raw_stat_columns:
            assert (result[f"{stat}_p10"] <= result[f"{stat}_p50"]).all(), (position, stat)
            assert (result[f"{stat}_p50"] <= result[f"{stat}_p90"]).all(), (position, stat)
